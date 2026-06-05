//! Architecture visualization: a self-contained SVG, hand-built by string
//! formatting (no deps), written once at training start as `viz.svg` in the
//! run dir. Promotion copies it into the model dir verbatim; the server
//! serves it on `/api/runs/{id}/viz` and `/api/models/{name}/viz`.

use std::path::Path;

use anyhow::Result;
use std::fmt::Write as _;

const BOX_W: f64 = 150.0;
const BOX_H: f64 = 86.0;
const GAP: f64 = 44.0;
const PAD: f64 = 24.0;
const BOX_Y: f64 = 46.0;
const H: f64 = 168.0;

#[allow(clippy::too_many_arguments)]
pub fn write_viz(
    path: &Path,
    n_pca: usize,
    n_meta: usize,
    channels: &[usize],
    kernel_size: usize,
    dense_hidden: usize,
    dropout: f64,
) -> Result<()> {
    std::fs::write(path, render(n_pca, n_meta, channels, kernel_size, dense_hidden, dropout))?;
    Ok(())
}

fn render(
    n_pca: usize,
    n_meta: usize,
    channels: &[usize],
    kernel_size: usize,
    dense_hidden: usize,
    dropout: f64,
) -> String {
    struct LayerBox {
        title: String,
        units: String,
        sub1: String,
        sub2: String,
    }

    let mut boxes = vec![LayerBox {
        title: "input".into(),
        units: (n_pca + n_meta).to_string(),
        sub1: format!("{n_pca} pca + {n_meta} meta"),
        sub2: "conv sees pca only".into(),
    }];
    let mut prev_ch = 1;
    // Track the post-pool signal length so each box can show its output
    // shape; mirrors the forward pass (pool halves, skipped at length < 2).
    let mut len = n_pca;
    for (i, &ch) in channels.iter().enumerate() {
        if len >= 2 {
            len /= 2;
        }
        boxes.push(LayerBox {
            title: format!("conv {}", i + 1),
            units: format!("{ch}×{len}"),
            sub1: format!("Conv1d {prev_ch}→{ch} k{kernel_size}"),
            sub2: format!("ReLU · pool/2 · drop {dropout}"),
        });
        prev_ch = ch;
    }
    boxes.push(LayerBox {
        title: "pool".into(),
        units: (prev_ch + n_meta).to_string(),
        sub1: format!("global avg → {prev_ch}"),
        sub2: format!("⊕ {n_meta} meta"),
    });
    boxes.push(LayerBox {
        title: "dense".into(),
        units: dense_hidden.to_string(),
        sub1: format!("Dense {}→{dense_hidden}", prev_ch + n_meta),
        sub2: format!("ReLU · dropout {dropout}"),
    });
    boxes.push(LayerBox {
        title: "output".into(),
        units: "1".into(),
        sub1: format!("Dense {dense_hidden}→1"),
        sub2: "linear".into(),
    });

    let n = boxes.len();
    let w = PAD * 2.0 + n as f64 * BOX_W + (n as f64 - 1.0) * GAP;
    let arch = channels.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" → ");

    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {H:.0}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace"><rect x="0" y="0" width="{w:.0}" height="{H:.0}" rx="6" fill="#fcfcfb"/><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L8,4L0,8z" fill="#8a8a8a"/></marker></defs><text x="{PAD:.0}" y="28" font-size="13" fill="#5a5a5a">CNN · {n_pca} pca → [{arch}] k{kernel_size} → {dense_hidden} → 1</text>"##
    );

    for (i, b) in boxes.iter().enumerate() {
        let x = PAD + i as f64 * (BOX_W + GAP);
        let cx = x + BOX_W / 2.0;
        let _ = write!(
            s,
            r##"<rect x="{x:.0}" y="{BOX_Y:.0}" width="{BOX_W:.0}" height="{BOX_H:.0}" rx="6" fill="#ffffff" stroke="#b9b6ae"/><text x="{cx:.0}" y="{ty:.0}" font-size="11" fill="#7a766c" text-anchor="middle">{title}</text><text x="{cx:.0}" y="{uy:.0}" font-size="22" fill="#2b2a26" text-anchor="middle">{units}</text>"##,
            ty = BOX_Y + 18.0,
            uy = BOX_Y + 44.0,
            title = b.title,
            units = b.units,
        );
        if !b.sub1.is_empty() {
            let _ = write!(
                s,
                r##"<text x="{cx:.0}" y="{y:.0}" font-size="10" fill="#7a766c" text-anchor="middle">{t}</text>"##,
                y = BOX_Y + 62.0,
                t = b.sub1,
            );
        }
        if !b.sub2.is_empty() {
            let _ = write!(
                s,
                r##"<text x="{cx:.0}" y="{y:.0}" font-size="10" fill="#7a766c" text-anchor="middle">{t}</text>"##,
                y = BOX_Y + 76.0,
                t = b.sub2,
            );
        }
        if i + 1 < n {
            let _ = write!(
                s,
                r##"<line x1="{x1:.0}" y1="{ym:.0}" x2="{x2:.0}" y2="{ym:.0}" stroke="#8a8a8a" stroke-width="1.25" marker-end="url(#arr)"/>"##,
                x1 = x + BOX_W + 4.0,
                x2 = x + BOX_W + GAP - 6.0,
                ym = BOX_Y + BOX_H / 2.0,
            );
        }
    }
    s.push_str("</svg>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_all_layers() {
        let svg = render(100, 41, &[32, 64], 3, 64, 0.1);
        assert!(svg.starts_with("<svg "));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("Conv1d 1→32 k3"));
        assert!(svg.contains("Conv1d 32→64 k3"));
        assert!(svg.contains("global avg → 64"));
        assert!(svg.contains("Dense 105→64"));
        assert!(svg.contains("Dense 64→1"));
        assert!(svg.contains("dropout 0.1"));
        // Post-pool lengths: 100 → 50 → 25.
        assert!(svg.contains("32×50"));
        assert!(svg.contains("64×25"));
    }
}
