use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use base64::prelude::*;
use clap::Parser;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_MODEL: &str = "qwen/qwen3-vl-30b-a3b-instruct";
const PALETTE: [&str; 11] = [
    "black", "white", "gray", "silver", "red", "orange", "yellow", "green", "blue", "purple",
    "brown",
];

#[derive(Parser, Debug)]
#[command(name = "vehicle-colour")]
#[command(about = "Classify vehicle colours with SAM masks and an OpenRouter VLM")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    images: PathBuf,

    #[arg(long)]
    out: PathBuf,

    /// Optional directory of SAM/AI segmentation masks. If omitted, a
    /// conservative bbox-prior mask is used.
    #[arg(long)]
    masks: Option<PathBuf>,

    /// OpenRouter API key. Defaults to OPENROUTER_API_KEY from .env/env.
    #[arg(long)]
    openrouter_api_key: Option<String>,

    /// OpenRouter model id. Defaults to MODEL from .env/env, then Qwen3-VL.
    #[arg(long)]
    model: Option<String>,
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

#[derive(Debug, Clone)]
enum SegmentationMask {
    External {
        image: image::GrayImage,
        origin_x: u32,
        origin_y: u32,
    },
    BoxPrior,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
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
                let dx = (nx - 0.5) / 0.47;
                let dy = (ny - 0.48) / 0.37;
                dx * dx + dy * dy <= 1.0 && (0.10..=0.82).contains(&ny)
            }
        }
    }
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let args = Args::parse();
    let api_key = args
        .openrouter_api_key
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .context("missing OpenRouter API key; pass --openrouter-api-key or set OPENROUTER_API_KEY in .env")?;
    let model = args
        .model
        .or_else(|| std::env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let client = Client::new();
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

        let masked_crop = build_masked_crop(&image_path, &detection.bbox_pixels, &mask)
            .with_context(|| format!("failed to build masked crop for {}", detection.image))?;
        let colour = classify_with_vlm(&client, &api_key, &model, &masked_crop)
            .with_context(|| format!("failed to classify {}", image_path.display()))?;

        detection.colour = Some(colour);
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

fn build_masked_crop(image_path: &Path, bbox: &BBox, mask: &SegmentationMask) -> Result<Vec<u8>> {
    let image = image::open(image_path)
        .with_context(|| format!("could not open image {}", image_path.display()))?
        .to_rgb8();
    let (image_w, image_h) = image.dimensions();

    let left = bbox.left.floor().max(0.0) as u32;
    let top = bbox.top.floor().max(0.0) as u32;
    let right = ((bbox.left + bbox.width).ceil().min(image_w as f64) as u32).max(left);
    let bottom = ((bbox.top + bbox.height).ceil().min(image_h as f64) as u32).max(top);

    if right <= left || bottom <= top {
        bail!("empty bounding box after clamping");
    }

    let crop_w = right - left;
    let crop_h = bottom - top;
    let mut crop = RgbaImage::from_pixel(crop_w, crop_h, Rgba([255, 255, 255, 255]));

    for y in top..bottom {
        for x in left..right {
            let nx = (x - left) as f64 / crop_w as f64;
            let ny = (y - top) as f64 / crop_h as f64;
            if mask.contains(x, y, nx, ny) {
                let [r, g, b] = image.get_pixel(x, y).0;
                crop.put_pixel(x - left, y - top, Rgba([r, g, b, 255]));
            }
        }
    }

    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(crop)
        .write_to(&mut encoded, ImageFormat::Png)
        .context("failed to encode masked crop as PNG")?;
    Ok(encoded.into_inner())
}

fn classify_with_vlm(
    client: &Client,
    api_key: &str,
    model: &str,
    masked_crop_png: &[u8],
) -> Result<String> {
    let image_url = format!(
        "data:image/png;base64,{}",
        BASE64_STANDARD.encode(masked_crop_png)
    );
    let palette = PALETTE.join(", ");
    let prompt = format!(
        "What is the likely exterior paint colour of this vehicle? \
Choose exactly one from this palette: {palette}. \
Ignore shadows, underexposure, windows, tyres, road, reflections, lights, licence plates, and masked white background. \
Return only the single lowercase colour word."
    );

    let response = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("HTTP-Referer", "https://localhost/vehicle-colour")
        .header("X-Title", "Vehicle Colour Classifier")
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        { "type": "image_url", "image_url": { "url": image_url } }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 16
        }))
        .send()
        .context("failed to send OpenRouter request")?;

    let status = response.status();
    let body = response
        .text()
        .context("failed to read OpenRouter response")?;
    if !status.is_success() {
        bail!("OpenRouter returned HTTP {status}: {body}");
    }

    let parsed: OpenRouterResponse =
        serde_json::from_str(&body).context("failed to parse OpenRouter response")?;
    let content = parsed
        .choices
        .first()
        .map(|choice| choice.message.content.trim())
        .context("OpenRouter response did not include a choice")?;

    normalize_colour(content)
        .with_context(|| format!("model returned non-palette colour: {content:?}"))
}

fn normalize_colour(content: &str) -> Result<String> {
    let lower = content.to_lowercase();
    let cleaned = lower
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphabetic());

    if cleaned == "grey" {
        return Ok("gray".to_string());
    }

    for colour in PALETTE {
        if cleaned == colour {
            return Ok(colour.to_string());
        }
    }

    for word in lower.split(|ch: char| !ch.is_ascii_alphabetic()) {
        if word == "grey" {
            return Ok("gray".to_string());
        }
        if PALETTE.contains(&word) {
            return Ok(word.to_string());
        }
    }

    bail!("no palette colour found")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_direct_palette_word() {
        assert_eq!(normalize_colour("silver").unwrap(), "silver");
        assert_eq!(normalize_colour("Gray.").unwrap(), "gray");
    }

    #[test]
    fn normalizes_short_explanatory_response() {
        assert_eq!(
            normalize_colour("The likely colour is silver.").unwrap(),
            "silver"
        );
    }
}
