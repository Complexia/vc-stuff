use std::cmp::max;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use image::Pixel;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "vehicle-colour")]
#[command(about = "Classify vehicle colours for bounding-box detections")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    images: PathBuf,

    #[arg(long)]
    out: PathBuf,

    /// Optional directory of AI segmentation masks. If omitted, a conservative
    /// bbox-prior mask is used so the CLI remains runnable without model weights.
    #[arg(long)]
    masks: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Detection {
    image: String,
    bbox_pixels: BBox,
    #[serde(skip_serializing_if = "Option::is_none")]
    colour: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct BBox {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    r: f64,
    g: f64,
    b: f64,
    h: f64,
    s: f64,
    v: f64,
    weight: f64,
}

#[derive(Debug, Clone, Copy)]
struct Aggregate {
    r: f64,
    g: f64,
    b: f64,
    h: f64,
    s: f64,
    v: f64,
}

#[derive(Debug, Clone)]
enum SegmentationMask {
    External {
        image: image::GrayImage,
        origin_x: u32,
        origin_y: u32,
    },
    BoxPrior,
}

impl SegmentationMask {
    fn contains(&self, image_x: u32, image_y: u32, nx: f64, ny: f64) -> bool {
        match self {
            SegmentationMask::External {
                image,
                origin_x,
                origin_y,
            } => {
                if image_x < *origin_x || image_y < *origin_y {
                    return false;
                }
                let mask_x = image_x - origin_x;
                let mask_y = image_y - origin_y;
                mask_x < image.width()
                    && mask_y < image.height()
                    && image.get_pixel(mask_x, mask_y).0[0] >= 128
            }
            SegmentationMask::BoxPrior => {
                // Coarse fallback for local runs without model weights. It keeps
                // the likely vehicle body center and rejects common bbox clutter.
                let dx = (nx - 0.5) / 0.47;
                let dy = (ny - 0.48) / 0.37;
                dx * dx + dy * dy <= 1.0 && (0.10..=0.82).contains(&ny)
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest = fs::read_to_string(&args.manifest)
        .with_context(|| format!("failed to read manifest {}", args.manifest.display()))?;
    let detections: Vec<Detection> =
        serde_json::from_str(&manifest).context("failed to parse manifest JSON")?;

    let mut per_image_seen = HashMap::<String, usize>::new();
    let mut predictions = Vec::with_capacity(detections.len());
    for (detection_index, mut detection) in detections.into_iter().enumerate() {
        let image_path = args.images.join(&detection.image);
        let per_image_index = per_image_seen.entry(detection.image.clone()).or_default();
        let mask = load_segmentation_mask(
            args.masks.as_deref(),
            &detection.image,
            detection_index,
            *per_image_index,
            &detection.bbox_pixels,
        )
        .with_context(|| format!("failed to load mask for {}", detection.image))?;
        *per_image_index += 1;

        let colour = classify_detection(&image_path, &detection.bbox_pixels, &mask)
            .with_context(|| format!("failed to classify {}", image_path.display()))?;
        detection.colour = Some(colour.to_string());
        predictions.push(detection);
    }

    let output =
        serde_json::to_string_pretty(&predictions).context("failed to encode output JSON")?;
    fs::write(&args.out, format!("{output}\n"))
        .with_context(|| format!("failed to write {}", args.out.display()))?;

    Ok(())
}

fn load_segmentation_mask(
    mask_dir: Option<&Path>,
    image_name: &str,
    detection_index: usize,
    per_image_index: usize,
    bbox: &BBox,
) -> Result<SegmentationMask> {
    let Some(mask_dir) = mask_dir else {
        return Ok(SegmentationMask::BoxPrior);
    };

    let image_stem = Path::new(image_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("image filename has no valid stem")?;
    let candidates = [
        mask_dir.join(format!("{detection_index}.png")),
        mask_dir.join(format!("{image_stem}__{detection_index}.png")),
        mask_dir.join(format!("{image_stem}__{per_image_index}.png")),
        mask_dir.join(format!("{image_stem}.png")),
    ];

    for candidate in candidates {
        if candidate.exists() {
            let mask_image = image::open(&candidate)
                .with_context(|| format!("could not open mask {}", candidate.display()))?
                .to_luma8();

            let full_image_mask =
                candidate.file_stem().and_then(|stem| stem.to_str()) == Some(image_stem);
            let (origin_x, origin_y) = if full_image_mask {
                (0, 0)
            } else {
                (
                    bbox.left.floor().max(0.0) as u32,
                    bbox.top.floor().max(0.0) as u32,
                )
            };

            return Ok(SegmentationMask::External {
                image: mask_image,
                origin_x,
                origin_y,
            });
        }
    }

    Ok(SegmentationMask::BoxPrior)
}

fn classify_detection(
    image_path: &Path,
    bbox: &BBox,
    mask: &SegmentationMask,
) -> Result<&'static str> {
    let image = image::open(image_path)
        .with_context(|| format!("could not open image {}", image_path.display()))?;
    let rgb = image.to_rgb8();
    let (image_w, image_h) = rgb.dimensions();
    let scales = gray_world_scales(&rgb);

    let left = bbox.left.floor().max(0.0) as i64;
    let top = bbox.top.floor().max(0.0) as i64;
    let right = (bbox.left + bbox.width).ceil().min(image_w as f64) as i64;
    let bottom = (bbox.top + bbox.height).ceil().min(image_h as f64) as i64;

    if right <= left || bottom <= top {
        bail!("empty bounding box after clamping");
    }

    let crop_w = (right - left) as u32;
    let crop_h = (bottom - top) as u32;
    let stride = max(1, max(crop_w, crop_h) / 180);
    let mut samples = Vec::new();

    for y in (top as u32..bottom as u32).step_by(stride as usize) {
        for x in (left as u32..right as u32).step_by(stride as usize) {
            let nx = (x as f64 - left as f64) / crop_w as f64;
            let ny = (y as f64 - top as f64) / crop_h as f64;

            if !(0.02..=0.98).contains(&nx) || !(0.04..=0.94).contains(&ny) {
                continue;
            }

            if !mask.contains(x, y, nx, ny) {
                continue;
            }

            let channels = rgb.get_pixel(x, y).to_rgb().0;
            let r = ((channels[0] as f64 / 255.0) * scales.0).min(1.0);
            let g = ((channels[1] as f64 / 255.0) * scales.1).min(1.0);
            let b = ((channels[2] as f64 / 255.0) * scales.2).min(1.0);
            let (h, s, v) = rgb_to_hsv(r, g, b);

            let center_weight = gaussian(nx, 0.5, 0.42) * gaussian(ny, 0.48, 0.34);
            let mut weight = 0.35 + center_weight;

            if matches!(mask, SegmentationMask::BoxPrior) && ny > 0.66 {
                weight *= 0.55;
            }
            if v < 0.08 || v > 0.97 {
                weight *= 0.15;
            }
            if s > 0.68 && (h < 30.0 || h > 330.0) && v > 0.45 {
                weight *= 0.25;
            }

            samples.push(Sample {
                r,
                g,
                b,
                h,
                s,
                v,
                weight,
            });
        }
    }

    if samples.is_empty() {
        bail!("no pixels sampled from bounding box");
    }

    let filtered = select_body_like_samples(&samples);
    Ok(classify_samples(&filtered))
}

fn select_body_like_samples(samples: &[Sample]) -> Vec<Sample> {
    let filtered: Vec<Sample> = samples
        .iter()
        .copied()
        .filter(|sample| {
            let neutral = sample.s < 0.26 && (0.16..=0.90).contains(&sample.v);
            let chromatic_body = sample.s >= 0.18 && sample.v >= 0.16 && sample.v <= 0.92;
            neutral || chromatic_body
        })
        .collect();

    if filtered.len() >= max(32, samples.len() / 5) {
        filtered
    } else {
        samples.to_vec()
    }
}

fn gray_world_scales(image: &image::RgbImage) -> (f64, f64, f64) {
    let (width, height) = image.dimensions();
    let stride = max(1, max(width, height) / 240);
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;
    let mut count = 0.0;

    for y in (0..height).step_by(stride as usize) {
        for x in (0..width).step_by(stride as usize) {
            let [r, g, b] = image.get_pixel(x, y).0;
            r_sum += r as f64 / 255.0;
            g_sum += g as f64 / 255.0;
            b_sum += b as f64 / 255.0;
            count += 1.0;
        }
    }

    let r_mean = r_sum / count;
    let g_mean = g_sum / count;
    let b_mean = b_sum / count;
    let gray = (r_mean + g_mean + b_mean) / 3.0;

    (
        (gray / r_mean.max(0.01)).clamp(0.65, 1.55),
        (gray / g_mean.max(0.01)).clamp(0.65, 1.55),
        (gray / b_mean.max(0.01)).clamp(0.65, 1.55),
    )
}

fn aggregate_samples(samples: &[Sample]) -> Aggregate {
    let mut wr = 0.0;
    let mut wg = 0.0;
    let mut wb = 0.0;
    let mut ws = 0.0;
    let mut wv = 0.0;
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut total = 0.0;

    for sample in samples {
        let mut w = sample.weight;
        if sample.s < 0.12 && sample.v > 0.82 {
            w *= 0.45;
        }
        if sample.v < 0.12 {
            w *= 0.55;
        }

        wr += sample.r * w;
        wg += sample.g * w;
        wb += sample.b * w;
        ws += sample.s * w;
        wv += sample.v * w;
        wx += sample.h.to_radians().cos() * sample.s * w;
        wy += sample.h.to_radians().sin() * sample.s * w;
        total += w;
    }

    let mut hue = wy.atan2(wx).to_degrees();
    if hue < 0.0 {
        hue += 360.0;
    }

    Aggregate {
        r: wr / total,
        g: wg / total,
        b: wb / total,
        h: hue,
        s: ws / total,
        v: wv / total,
    }
}

fn classify_samples(samples: &[Sample]) -> &'static str {
    let mut votes = [0.0_f64; 11];

    for sample in samples {
        let label = classify_sample(*sample);
        votes[label_index(label)] += sample.weight * sample_confidence(*sample);
    }

    let mut best_idx = 0;
    for idx in 1..votes.len() {
        if votes[idx] > votes[best_idx] {
            best_idx = idx;
        }
    }

    let aggregate = aggregate_samples(samples);
    let aggregate_label = classify_aggregate(aggregate);
    if aggregate_label == "black" || label_for_index(best_idx) == "black" {
        if let Some(neutral_label) = underexposed_neutral_label(samples, aggregate) {
            return neutral_label;
        }
    }
    let aggregate_idx = label_index(aggregate_label);

    if votes[aggregate_idx] >= votes[best_idx] * 0.82 {
        aggregate_label
    } else {
        label_for_index(best_idx)
    }
}

fn underexposed_neutral_label(samples: &[Sample], aggregate: Aggregate) -> Option<&'static str> {
    if aggregate.s > 0.24 || aggregate.v > 0.24 {
        return None;
    }

    let mut neutral_luminance: Vec<f64> = samples
        .iter()
        .filter(|sample| sample.s < 0.30)
        .map(|sample| 0.2126 * sample.r + 0.7152 * sample.g + 0.0722 * sample.b)
        .collect();
    if neutral_luminance.len() < 32 {
        return None;
    }

    neutral_luminance.sort_by(|a, b| a.total_cmp(b));
    let p75 = percentile(&neutral_luminance, 0.75);
    let p90 = percentile(&neutral_luminance, 0.90);

    if p75 > 0.42 || p90 > 0.56 {
        Some("silver")
    } else if p75 > 0.18 || p90 > 0.22 {
        Some("gray")
    } else {
        None
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn classify_sample(sample: Sample) -> &'static str {
    if sample.v < 0.16 {
        return "black";
    }

    let luminance = 0.2126 * sample.r + 0.7152 * sample.g + 0.0722 * sample.b;
    if luminance < 0.15 {
        return "black";
    }

    if sample.s < 0.25 {
        return neutral_label(sample.v, luminance);
    }

    if sample.s < 0.38 {
        if (15.0..=45.0).contains(&sample.h) && sample.v < 0.46 {
            return "brown";
        }
        return neutral_label(sample.v, luminance);
    }

    if (165.0..255.0).contains(&sample.h) && (sample.s < 0.70 || sample.v < 0.55) {
        return neutral_label(sample.v, luminance);
    }

    classify_aggregate(Aggregate {
        r: sample.r,
        g: sample.g,
        b: sample.b,
        h: sample.h,
        s: sample.s,
        v: sample.v,
    })
}

fn neutral_label(value: f64, luminance: f64) -> &'static str {
    if value > 0.80 && luminance > 0.72 {
        "white"
    } else if value > 0.48 {
        "silver"
    } else if value > 0.22 {
        "gray"
    } else {
        "black"
    }
}

fn sample_confidence(sample: Sample) -> f64 {
    if sample.s < 0.25 {
        0.9
    } else if sample.s > 0.48 {
        1.15
    } else {
        1.0
    }
}

fn classify_aggregate(c: Aggregate) -> &'static str {
    let luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

    if c.v < 0.18 || luminance < 0.16 {
        return "black";
    }

    if c.s < 0.18 {
        if c.v > 0.78 && luminance > 0.72 {
            return "white";
        }
        if c.v > 0.46 {
            return "silver";
        }
        return "gray";
    }

    if c.s < 0.28 && c.v > 0.52 {
        return "silver";
    }

    if (165.0..255.0).contains(&c.h) && (c.s < 0.62 || c.v < 0.52) {
        return neutral_label(c.v, luminance);
    }

    let h = c.h;
    if h >= 345.0 || h < 12.0 {
        if c.v < 0.38 || c.s < 0.38 {
            "brown"
        } else {
            "red"
        }
    } else if h < 34.0 {
        if c.v < 0.58 || c.s < 0.45 {
            "brown"
        } else {
            "orange"
        }
    } else if h < 68.0 {
        if c.v < 0.42 {
            "brown"
        } else {
            "yellow"
        }
    } else if h < 165.0 {
        "green"
    } else if h < 255.0 {
        "blue"
    } else if h < 310.0 {
        "purple"
    } else if c.v < 0.45 {
        "brown"
    } else {
        "red"
    }
}

fn label_index(label: &str) -> usize {
    match label {
        "black" => 0,
        "white" => 1,
        "gray" => 2,
        "silver" => 3,
        "red" => 4,
        "orange" => 5,
        "yellow" => 6,
        "green" => 7,
        "blue" => 8,
        "purple" => 9,
        "brown" => 10,
        _ => unreachable!("unknown label"),
    }
}

fn label_for_index(index: usize) -> &'static str {
    match index {
        0 => "black",
        1 => "white",
        2 => "gray",
        3 => "silver",
        4 => "red",
        5 => "orange",
        6 => "yellow",
        7 => "green",
        8 => "blue",
        9 => "purple",
        10 => "brown",
        _ => unreachable!("unknown label index"),
    }
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    let delta = max_c - min_c;

    let hue = if delta == 0.0 {
        0.0
    } else if max_c == r {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if max_c == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let saturation = if max_c == 0.0 { 0.0 } else { delta / max_c };
    (hue, saturation, max_c)
}

fn gaussian(value: f64, mean: f64, sigma: f64) -> f64 {
    let z = (value - mean) / sigma;
    (-0.5 * z * z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_colours_snap_to_expected_palette() {
        assert_eq!(
            classify_aggregate(Aggregate {
                r: 0.08,
                g: 0.08,
                b: 0.08,
                h: 0.0,
                s: 0.0,
                v: 0.08
            }),
            "black"
        );
        assert_eq!(
            classify_aggregate(Aggregate {
                r: 0.72,
                g: 0.72,
                b: 0.70,
                h: 55.0,
                s: 0.03,
                v: 0.72
            }),
            "silver"
        );
    }

    #[test]
    fn hue_ranges_cover_common_car_colours() {
        assert_eq!(classify_hsv(0.0, 0.8, 0.7), "red");
        assert_eq!(classify_hsv(28.0, 0.8, 0.7), "orange");
        assert_eq!(classify_hsv(52.0, 0.8, 0.7), "yellow");
        assert_eq!(classify_hsv(120.0, 0.8, 0.7), "green");
        assert_eq!(classify_hsv(220.0, 0.8, 0.7), "blue");
        assert_eq!(classify_hsv(280.0, 0.8, 0.7), "purple");
        assert_eq!(classify_hsv(25.0, 0.35, 0.35), "brown");
    }

    fn classify_hsv(h: f64, s: f64, v: f64) -> &'static str {
        classify_aggregate(Aggregate {
            r: v,
            g: v,
            b: v,
            h,
            s,
            v,
        })
    }
}
