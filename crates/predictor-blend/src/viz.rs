//! Architecture SVG: member boxes fanning into the combiner. Written at
//! train start per the visualization contract; ArchViz has no native blend
//! family, so this is what the UI shows.

use crate::spec::Rule;
use crate::Resolved;

pub fn render(members: &[Resolved], rule: Rule) -> String {
    const ROW_H: f32 = 44.0;
    const BOX_W: f32 = 240.0;
    let n = members.len();
    let height = (n as f32 * (ROW_H + 12.0) + 60.0).max(140.0);
    let mid_y = height / 2.0;
    let comb_x = BOX_W + 90.0;
    let rule_label = match rule {
        Rule::Mean => "weighted mean",
        Rule::Median => "median vote",
    };

    let mut rows = String::new();
    for (i, m) in members.iter().enumerate() {
        let y = 30.0 + i as f32 * (ROW_H + 12.0);
        let title = match &m.source {
            Some(s) => s.clone(),
            None => m.predictor.name.clone(),
        };
        let sub = format!(
            "{} · {} cols{}",
            if m.frozen { "frozen" } else { "trained" },
            m.col_idx.len(),
            if m.exclude_blocks.is_empty() {
                String::new()
            } else {
                format!(" · −{}", m.exclude_blocks.join(",−"))
            }
        );
        rows.push_str(&format!(
            r##"<rect x="8" y="{y}" width="{BOX_W}" height="{ROW_H}" rx="6" fill="#f6f7f9" stroke="#6b7280"/>
<text x="20" y="{ty}" font-family="ui-monospace,monospace" font-size="13" fill="#111827">{title}</text>
<text x="20" y="{sy}" font-family="ui-monospace,monospace" font-size="10" fill="#6b7280">{sub}</text>
<path d="M {bx} {by} C {cx1} {by}, {cx1} {mid_y}, {comb_x} {mid_y}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>
"##,
            ty = y + 18.0,
            sy = y + 33.0,
            bx = 8.0 + BOX_W,
            by = y + ROW_H / 2.0,
            cx1 = 8.0 + BOX_W + 45.0,
        ));
    }

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {height}" width="{w}" height="{height}">
{rows}<circle cx="{cx}" cy="{mid_y}" r="26" fill="#fff7ed" stroke="#374151" stroke-width="1.5"/>
<text x="{cx}" y="{cy1}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="11" fill="#111827">Σ</text>
<text x="{cx}" y="{cy2}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9" fill="#6b7280">{rule_label}</text>
<path d="M {ax} {mid_y} h 50" fill="none" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#a)"/>
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/></marker></defs>
<text x="{px}" y="{py}" font-family="ui-monospace,monospace" font-size="12" fill="#111827">price</text>
</svg>"##,
        w = comb_x + 140.0,
        cx = comb_x + 26.0,
        cy1 = mid_y - 2.0,
        cy2 = mid_y + 11.0,
        ax = comb_x + 52.0,
        px = comb_x + 108.0,
        py = mid_y + 4.0,
    )
}
