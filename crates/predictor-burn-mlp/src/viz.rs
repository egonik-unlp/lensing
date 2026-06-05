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

pub fn write_viz(
    path: &Path,
    n_in: usize,
    hidden: &[usize],
    dropout: f64,
    activation: &str,
) -> Result<()> {
    std::fs::write(path, render(n_in, hidden, dropout, activation))?;
    Ok(())
}

fn render(n_in: usize, hidden: &[usize], dropout: f64, activation: &str) -> String {
    struct LayerBox {
        title: String,
        units: String,
        sub1: String,
        sub2: String,
    }

    let mut boxes = vec![LayerBox {
        title: "input".into(),
        units: n_in.to_string(),
        sub1: "features".into(),
        sub2: String::new(),
    }];
    let mut prev = n_in;
    for (i, &h) in hidden.iter().enumerate() {
        boxes.push(LayerBox {
            title: format!("hidden {}", i + 1),
            units: h.to_string(),
            sub1: format!("Dense {prev}→{h}"),
            sub2: format!("{activation} · dropout {dropout}"),
        });
        prev = h;
    }
    boxes.push(LayerBox {
        title: "output".into(),
        units: "1".into(),
        sub1: format!("Dense {prev}→1"),
        sub2: "linear".into(),
    });

    let n = boxes.len();
    let w = PAD * 2.0 + n as f64 * BOX_W + (n as f64 - 1.0) * GAP;
    let arch = hidden.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(" → ");

    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {H:.0}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace"><rect x="0" y="0" width="{w:.0}" height="{H:.0}" rx="6" fill="#fcfcfb"/><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L8,4L0,8z" fill="#8a8a8a"/></marker></defs><text x="{PAD:.0}" y="28" font-size="13" fill="#5a5a5a">MLP · {n_in} → {arch} → 1</text>"##
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
        let svg = render(83, &[256, 128], 0.1, "GELU");
        assert!(svg.starts_with("<svg "));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("Dense 83→256"));
        assert!(svg.contains("Dense 256→128"));
        assert!(svg.contains("Dense 128→1"));
        assert!(svg.contains("GELU · dropout 0.1"));
    }
}
