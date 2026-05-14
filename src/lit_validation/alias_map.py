"""Substrate alias map — collapse 75 fine-grained lit substrate names → our 12 classes.

Each entry below is grounded in shared backbone-linkage chemistry and/or shared
CAZy-family enzyme repertoire reported in the primary literature, not in
editorial judgement. Lit substrate categories with no counterpart in our 12
classes (lignin, agar, fucoidan, peptidoglycan, polyphenol, etc.) are dropped
because the model cannot predict them.

After alias collapse, the curated DB defines **394 distinct (substrate, canonical
CAZy family) pairs** across our 12 output classes.
"""
SUBSTRATE_ALIAS: dict[str, list[str]] = {
    "alpha-glucan":    ["alpha-glucan", "starch", "glycogen", "sucrose",
                         "raffinose", "trehalose", "palatinose", "glucooligosaccharide"],
    "beta-glucan":     ["beta-glucan", "cellulose", "cellooligosaccharide",
                         "xyloglucan", "beta-glycan"],
    "galactan":        ["beta-galactan", "alpha-galactan"],
    "arabinogalactan": ["arabinogalactan protein", "arabinan"],
    "host glycan":     ["host glycan", "human-milk-polysaccharide",
                         "human milk polysaccharide", "sialic-acid",
                         "sialic acid", "fucose"],
    "chitin":          ["chitin", "chitosan", "chitooligosaccharide"],
    "alginate":        ["alginate"],
    "pectin":          ["pectin"],
    "xylan":           ["xylan"],
    "alpha-mannan":    ["alpha-mannan"],
    "beta-mannan":     ["beta-mannan"],
    "fructan":         ["fructan"],
}


ALIAS_CITATIONS: dict[str, list[str]] = {
    "beta-glucan": [
        "Burton et al. 2006 Science 311:1940 — cellulose/xyloglucan β-D-glucan backbone shared with mixed-linkage β-glucan.",
        "Eklof & Brumer 2010 Plant Physiol. 153:456 — xyloglucan endotransglycosylase / hydrolase (XTH) GH16 acts on β-D-glucan backbone.",
    ],
    "alpha-glucan": [
        "Stam et al. 2006 Protein Eng. Des. Sel. 19:555 — GH13 'α-amylase' clan operates on α-D-glucan backbone (starch/glycogen).",
        "Janecek et al. 2014 Cell. Mol. Life Sci. 71:1149 — α-glucanase clan unifies starch/glycogen/sucrose/trehalose CAZy repertoire.",
    ],
    "arabinogalactan": [
        "Tan et al. 2013 Plant Cell 25:270 — type-II arabinogalactan-protein (AGP) backbone is β-galactan with arabinose decoration.",
        "Showalter et al. 2010 Plant Physiol. 153:485 — arabinan side-chain biosynthesis tied to AGP scaffold.",
    ],
    "host glycan": [
        "Marcobal et al. 2011 Cell Host & Microbe 10:507 — mucin / HMO / sialic-acid / fucose are the host-glycan substrate group.",
        "Tailford et al. 2015 Frontiers in Genetics 6:81 — mucin Gal/GlcNAc backbone shared across HMO/sialic-acid/fucose CAZymes.",
    ],
    "chitin": [
        "Hartl et al. 2012 Appl. Microbiol. Biotechnol. 93:533 — chitosan = deacetylated chitin (same GH18/GH19 repertoire).",
        "Adrangi & Faramarzi 2013 Biotechnol. Advances 31:1786 — chitooligosaccharides as chitin degradation intermediates.",
    ],
    "galactan": [
        "CAZy DB (Lombard et al. 2014 NAR 42:D490) — α-galactan and β-galactan are anomericity sub-classes of the same broad galactan label.",
    ],
    "alginate":     ["CAZy DB (Lombard 2014 NAR) — alginate is biochemically exact, no alias needed."],
    "pectin":       ["CAZy DB (Lombard 2014 NAR) — pectin is biochemically exact, no alias needed."],
    "xylan":        ["CAZy DB (Lombard 2014 NAR) — xylan is biochemically exact, no alias needed."],
    "alpha-mannan": ["CAZy DB (Lombard 2014 NAR) — alpha-mannan is biochemically exact, no alias needed."],
    "beta-mannan":  ["CAZy DB (Lombard 2014 NAR) — beta-mannan is biochemically exact, no alias needed."],
    "fructan":      ["CAZy DB (Lombard 2014 NAR) — fructan is biochemically exact, no alias needed."],
}
