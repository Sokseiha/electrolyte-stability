#!/usr/bin/env python
"""Draw atom-mapped elementary steps as 2D reaction schemes.

Takes lines in the dataset format used by this repo:

    mapped_reactant>>mapped_product[|sequence_idx]

Fragments on each side are laid out on a single line separated by '+', in the
conventional way a reaction scheme is written. A whole mechanism (all steps
sharing a sequence_idx) is drawn as one image: A -> B -> C ..., wrapping onto
several rows when it is too wide.

Examples
--------
    # a single reaction passed directly
    python examples/draw_reaction.py "[Cl:1][S:2]...>>[Cl:1][S:2]..." -o rxn.png

    # line 3 of a dataset file (1-indexed)
    python examples/draw_reaction.py --file data/test.txt --line 3

    # a whole mechanism as one multi-arrow image
    python examples/draw_reaction.py --file data/test.txt --seq 11831 -o mech.png

    # keep explicit hydrogens, show atom maps, mark the reacting centre
    python examples/draw_reaction.py "<rxn>" --keep-hs --maps --highlight
"""

import argparse
import io
import os
import sys

# repo root on the path so `utils` is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageChops, ImageDraw, ImageFont

RDLogger.DisableLog("rdApp.*")

# same parser settings the repo uses everywhere (explicit Hs preserved)
ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

COLOR_CHANGED = (1.0, 0.55, 0.0)     # orange: electron count changed here
BOND_LENGTH = 28                     # px, fixed so every fragment draws to scale
GAP = 18                             # px between a fragment and its '+'
ROW_GAP = 40                         # px between wrapped rows

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse_line(line):
    """Split a dataset line into (reactant_smiles, product_smiles, seq_idx)."""
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    seq_idx = None
    if "|" in line:
        parts = line.split("|")
        line, seq_idx = parts[0], parts[1]
    if ">>" not in line:
        raise ValueError("no '>>' found - not a reaction line")
    reac, prod = line.split(">>")
    return reac.strip(), prod.strip(), seq_idx


def canonical(smi):
    """Canonical form used to test whether two species are the same."""
    mol = Chem.MolFromSmiles(smi, ps)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def changed_sites(reac_smi, prod_smi):
    """Atom-map numbers / map-pairs whose BE matrix entries differ.

    Empty sets if the matrices cannot be built (e.g. unmapped SMILES).
    """
    try:
        from utils.data_utils import get_BE_matrix
        reac_be = get_BE_matrix(reac_smi)
        prod_be = get_BE_matrix(prod_smi)
    except Exception as e:
        print(f"[warn] could not build BE matrices, no highlights: {e}",
              file=sys.stderr)
        return set(), set()

    if reac_be.shape != prod_be.shape:
        print("[warn] reactant/product BE shapes differ, skipping highlights",
              file=sys.stderr)
        return set(), set()

    diff = prod_be - reac_be
    n = diff.shape[0]
    atoms, bonds = set(), set()
    for i in range(n):
        if diff[i, i] != 0:                       # non-bonding electrons changed
            atoms.add(i + 1)                      # BE index 0-based -> map 1-based
        for j in range(i + 1, n):
            if diff[i, j] != 0:                   # bond order changed
                bonds.add((i + 1, j + 1))
                atoms.add(i + 1)
                atoms.add(j + 1)
    return atoms, bonds


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #

def autocrop(img, pad=6):
    """Trim white margins, then re-add a small uniform padding."""
    bg = Image.new(img.mode, img.size, "white")
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:                              # completely blank
        return img
    img = img.crop(bbox)
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    out.paste(img, (pad, pad))
    return out


def hstack(images, gap=0, bg="white"):
    """Concatenate images left-to-right, vertically centred."""
    images = [im for im in images if im is not None]
    if not images:
        return Image.new("RGB", (1, 1), bg)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    h = max(im.height for im in images)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        out.paste(im, (x, (h - im.height) // 2))
        x += im.width + gap
    return out


def vstack(images, gap=0, bg="white"):
    """Stack images top-to-bottom, left-aligned."""
    images = [im for im in images if im is not None]
    if not images:
        return Image.new("RGB", (1, 1), bg)
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + gap
    return out


def text_image(text, size=34, color="black"):
    """A tight image containing just `text` (used for '+' separators)."""
    font = get_font(size)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = tmp.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    img = Image.new("RGB", (w + 10, h + 10), "white")
    ImageDraw.Draw(img).text((5 - box[0], 5 - box[1]), text, font=font, fill=color)
    return img


def make_arrow(label=None, width=110, height=70):
    """A reaction arrow, optionally captioned above the shaft."""
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    y = height // 2
    x0, x1 = 8, width - 8
    d.line([(x0, y), (x1, y)], fill="black", width=3)
    head = 11
    d.polygon([(x1, y), (x1 - head, y - head // 2), (x1 - head, y + head // 2)],
              fill="black")
    if label:
        font = get_font(17)
        box = d.textbbox((0, 0), label, font=font)
        d.text(((width - (box[2] - box[0])) // 2, y - 26), label,
               font=font, fill="black")
    return img


# --------------------------------------------------------------------------- #
# molecule drawing
# --------------------------------------------------------------------------- #

def prepare_mol(smi, keep_hs, show_maps):
    """Parse a (multi-fragment) SMILES and return a mol ready to draw."""
    mol = Chem.MolFromSmiles(smi, ps)
    if mol is None:
        raise ValueError(f"RDKit could not parse: {smi[:80]}...")

    if not keep_hs:
        # explicit Hs here carry atom maps, so removeMapped must be enabled
        params = Chem.rdmolops.RemoveHsParameters()
        params.removeMapped = True
        # keep isolated hydrogens: a bare [H+] is a real species in mechanism
        # data, and removing it would silently drop it from the drawing
        params.removeDegreeZero = False
        mol = Chem.rdmolops.RemoveHs(mol, params, sanitize=True)

    # stash map numbers before the visible labels are (optionally) cleared,
    # so highlights can still be located afterwards
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetIntProp("origMapNum", atom.GetAtomMapNum())
    if not show_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return mol


def draw_fragment(frag, changed_atoms, changed_bonds):
    """Render one connected fragment to a tightly-cropped image."""
    Chem.rdDepictor.Compute2DCoords(frag)

    map2idx = {}
    for atom in frag.GetAtoms():
        m = atom.GetIntProp("origMapNum") if atom.HasProp("origMapNum") \
            else atom.GetAtomMapNum()
        if m:
            map2idx[m] = atom.GetIdx()

    hl_atoms = [map2idx[m] for m in changed_atoms if m in map2idx]
    hl_bonds = []
    for a, b in changed_bonds:
        if a in map2idx and b in map2idx:
            bond = frag.GetBondBetweenAtoms(map2idx[a], map2idx[b])
            if bond is not None:                  # bond may exist on one side only
                hl_bonds.append(bond.GetIdx())

    # canvas generous enough for the fragment at the fixed bond length; the
    # surplus is cropped away afterwards
    n = frag.GetNumAtoms()
    side = int(min(2600, max(320, 190 + 78 * n ** 0.62)))

    drawer = rdMolDraw2D.MolDraw2DCairo(side, side)
    opts = drawer.drawOptions()
    opts.fixedBondLength = BOND_LENGTH            # uniform scale across fragments
    opts.addStereoAnnotation = False
    opts.padding = 0.06
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, frag,
        highlightAtoms=hl_atoms,
        highlightBonds=hl_bonds,
        highlightAtomColors={i: COLOR_CHANGED for i in hl_atoms},
        highlightBondColors={i: COLOR_CHANGED for i in hl_bonds},
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    return autocrop(img), len(hl_atoms), len(hl_bonds)


def frag_map_nums(frag):
    """Atom map numbers present in a fragment."""
    out = set()
    for atom in frag.GetAtoms():
        m = atom.GetIntProp("origMapNum") if atom.HasProp("origMapNum") \
            else atom.GetAtomMapNum()
        if m:
            out.add(m)
    return out


def draw_side(smi, keep_hs, show_maps, changed_atoms=(), changed_bonds=(),
              reacting_maps=None):
    """Draw one side of a reaction: all fragments on one line, joined by '+'.

    If `reacting_maps` is given, fragments containing none of those atoms are
    spectators (solvent, counterions) and are omitted.
    """
    mol = prepare_mol(smi, keep_hs, show_maps)
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))

    n_dropped = 0
    if reacting_maps:
        keep = [f for f in frags if frag_map_nums(f) & reacting_maps]
        n_dropped = len(frags) - len(keep)
        if keep:                                  # never drop everything
            frags = keep

    parts, n_at, n_bd = [], 0, 0
    for i, frag in enumerate(frags):
        if i:
            parts.append(text_image("+"))
        img, a, b = draw_fragment(frag, changed_atoms, changed_bonds)
        parts.append(img)
        n_at += a
        n_bd += b
    return hstack(parts, gap=GAP), n_at, n_bd, n_dropped


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

def wrap_rows(elements, max_width):
    """Pack [species, arrow, species, arrow, ...] into rows under max_width.

    A row is broken after an arrow so the arrow signals continuation.
    """
    rows, row, width = [], [], 0
    for el in elements:
        w = el.width + GAP
        if row and width + w > max_width:
            # break after the trailing arrow if the row ends on one
            rows.append(row)
            row, width = [], 0
        row.append(el)
        width += w
    if row:
        rows.append(row)
    return [hstack(r, gap=GAP) for r in rows]


def render_single(line, out_path, keep_hs, show_maps, highlight, max_width,
                  hide_spectators=False):
    reac_smi, prod_smi, seq_idx = parse_line(line)
    ca, cb = changed_sites(reac_smi, prod_smi) if highlight else (set(), set())

    reacting = None
    if hide_spectators:
        reacting, _ = changed_sites(reac_smi, prod_smi)
    left, la, lb, ld = draw_side(reac_smi, keep_hs, show_maps, ca, cb, reacting)
    right, ra, rb, rd = draw_side(prod_smi, keep_hs, show_maps, ca, cb, reacting)
    canvas = hstack([left, make_arrow(), right], gap=GAP)
    canvas.save(out_path)

    msg = (f"wrote {out_path}  [{canvas.width}x{canvas.height}px]"
           + (f"  (sequence {seq_idx})" if seq_idx else ""))
    if hide_spectators and max(ld, rd):
        msg += f"  ({max(ld, rd)} spectator fragment(s) hidden)"
    if highlight:
        msg += f"  [BE changes: {len(ca)} atoms, {len(cb)} bonds]"
        if not keep_hs and len(ca) > max(la, ra):
            msg += "\n  note: some changed sites are on hydrogens; use --keep-hs"
    print(msg)
    return out_path


def build_pathways(steps, max_paths):
    """Turn the steps of one sequence_idx into root -> terminal pathways.

    A sequence_idx is a mechanistic *network*, not necessarily a linear chain:
    one starting reactant, branching intermediates, and one or more terminal
    products marked by self-loop steps. Each returned pathway is a list of
    mapped SMILES to be drawn left to right.
    """
    import networkx as nx

    repr_smi = {}                                 # canonical -> mapped SMILES
    graph = nx.DiGraph()
    for reac, prod, _ in steps:
        cr, cp = canonical(reac), canonical(prod)
        if cr is None or cp is None:
            continue
        repr_smi.setdefault(cr, reac)
        repr_smi.setdefault(cp, prod)
        graph.add_edge(cr, cp)

    terminals = set(nx.nodes_with_selfloops(graph))
    roots = [n for n, d in graph.in_degree() if d == 0]

    acyclic = graph.copy()
    acyclic.remove_edges_from(nx.selfloop_edges(acyclic))

    if len(roots) != 1 or not terminals:
        return None, graph, roots, terminals       # caller falls back

    # iterate terminals in a fixed order: set iteration depends on string
    # hashing, which Python randomises per process, so this would otherwise
    # renumber the pathways on every run
    paths = []
    for term in sorted(terminals):
        if term in acyclic:
            paths.extend(nx.all_simple_paths(acyclic, roots[0], term))
    if not paths:
        return None, graph, roots, terminals

    # total order: shortest first, ties broken by the species themselves
    paths.sort(key=lambda p: (len(p), p))
    if len(paths) > max_paths:
        print(f"[warn] {len(paths)} pathways found; drawing the {max_paths} "
              "shortest (raise --max-paths to see more)", file=sys.stderr)
        paths = paths[:max_paths]
    return [[repr_smi[c] for c in p] for p in paths], graph, roots, terminals


def render_chain(chain, keep_hs, show_maps, highlight, max_width, number_steps,
                 reacting_maps=None):
    """Render one linear species chain into wrapped row images."""
    elements = []
    n_at = n_bd = n_drop = 0
    for k, smi in enumerate(chain):
        # highlight what is about to change on the way out of this species
        if highlight and k < len(chain) - 1:
            ca, cb = changed_sites(smi, chain[k + 1])
        else:
            ca, cb = set(), set()
        img, a, b, d = draw_side(smi, keep_hs, show_maps, ca, cb, reacting_maps)
        n_at += a
        n_bd += b
        n_drop += d
        elements.append(img)
        if k < len(chain) - 1:
            elements.append(
                make_arrow(label=f"step {k + 1}" if number_steps else None))
    return wrap_rows(elements, max_width), n_at, n_bd, n_drop


def render_mechanism(lines, out_path, keep_hs, show_maps, highlight, max_width,
                     number_steps=True, max_paths=8, hide_spectators=False,
                     spectator_scope="sequence"):
    """Draw all steps sharing a sequence_idx as one multi-arrow image."""
    steps = [parse_line(l) for l in lines]
    seq_idx = steps[0][2]

    n_loops = sum(1 for r, p, _ in steps
                  if canonical(r) is not None and canonical(r) == canonical(p))

    # Atoms that take part in a reaction; everything else is a spectator
    # (solvent, counterion) carried along only to keep atom indices stable.
    #
    # Scope matters when the mechanism branches. "sequence" pools every step,
    # so a species that reacts in *any* pathway is drawn in *all* of them --
    # the molecule sets stay comparable across pathways, at the cost of showing
    # some species that are inert in the pathway you are looking at.
    # "pathway" decides per pathway, giving the least cluttered picture of each
    # one, but different rows may then show different species.
    seq_reacting = None
    if hide_spectators and spectator_scope == "sequence":
        seq_reacting = set()
        for reac, prod, _ in steps:
            ca, _cb = changed_sites(reac, prod)
            seq_reacting |= ca

    pathways, graph, roots, terminals = build_pathways(steps, max_paths)

    if pathways is None:
        # not a clean single-root network: fall back to file-order chains
        print(f"[warn] sequence {seq_idx} has {len(roots)} root(s) and "
              f"{len(terminals)} terminal(s); drawing in file order",
              file=sys.stderr)
        kept = [(r, p) for r, p, _ in steps if canonical(r) != canonical(p)]
        if not kept:
            raise ValueError("every step is a self-loop; nothing to draw")
        chains, cur = [], [kept[0][0], kept[0][1]]
        for i in range(1, len(kept)):
            if canonical(kept[i - 1][1]) == canonical(kept[i][0]):
                cur.append(kept[i][1])
            else:
                chains.append(cur)
                cur = [kept[i][0], kept[i][1]]
        chains.append(cur)
        pathways = chains

    blocks = []
    total_at = total_bd = total_drop = 0
    multi = len(pathways) > 1
    for i, chain in enumerate(pathways, start=1):
        if multi:
            # name the pathway by where it ends: several pathways of a
            # mechanism often differ only in which species keeps a proton
            end = canonical(chain[-1]) or ""
            if len(end) > 72:
                end = end[:69] + "..."
            blocks.append(text_image(f"Pathway {i} of {len(pathways)}"
                                     f"   →  {end}", size=19))

        if not hide_spectators:
            reacting_maps = None
        elif spectator_scope == "pathway":
            # only what actually reacts along *this* pathway
            reacting_maps = set()
            for k in range(len(chain) - 1):
                ca, _cb = changed_sites(chain[k], chain[k + 1])
                reacting_maps |= ca
        else:
            reacting_maps = seq_reacting

        rows, a, b, d = render_chain(chain, keep_hs, show_maps, highlight,
                                     max_width, number_steps, reacting_maps)
        blocks.extend(rows)
        total_at += a
        total_bd += b
        total_drop += d

    canvas = vstack(blocks, gap=ROW_GAP)
    canvas.save(out_path)

    msg = (f"wrote {out_path}  [{canvas.width}x{canvas.height}px]  "
           f"(sequence {seq_idx}): {len(steps) - n_loops} step(s), "
           f"{len(pathways)} pathway(s) from 1 start to "
           f"{len(terminals)} terminal product(s)")
    if n_loops:
        msg += f"; {n_loops} self-loop terminator(s) omitted"
    if hide_spectators:
        msg += (f"; {total_drop} spectator fragment slot(s) hidden "
                f"({spectator_scope} scope)")
    if highlight:
        msg += f"; {total_at} atoms / {total_bd} bonds highlighted"
    print(msg)
    if canvas.height > 4000 and not hide_spectators:
        print("  tip: --hide-spectators drops solvent/counterions and usually "
              "shrinks this a lot", file=sys.stderr)
    return out_path


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input (pick one)")
    src.add_argument("reaction", nargs="?",
                     help="reaction SMILES 'reactant>>product', optionally with |seq_idx")
    src.add_argument("--file", help="dataset file to read from")
    src.add_argument("--line", type=int,
                     help="with --file: 1-indexed line number to draw")
    src.add_argument("--seq",
                     help="with --file: draw every step with this sequence_idx "
                          "as a single multi-arrow image")

    ap.add_argument("-o", "--out", default="reaction.png", help="output PNG")
    ap.add_argument("--keep-hs", action="store_true",
                    help="keep explicit hydrogens (default: strip for readability)")
    ap.add_argument("--maps", action="store_true", help="show atom map numbers")
    ap.add_argument("--highlight", action="store_true",
                    help="highlight atoms/bonds whose electron count changes "
                         "(default: off)")
    ap.add_argument("--no-step-numbers", action="store_true",
                    help="with --seq: do not caption the arrows")
    ap.add_argument("--max-width", type=int, default=2000,
                    help="wrap onto a new row beyond this width in px (default: 2000)")
    ap.add_argument("--hide-spectators", action="store_true",
                    help="omit fragments that never react (solvent, counterions); "
                         "they are carried in every step only to keep atom "
                         "indices stable")
    ap.add_argument("--spectator-scope", choices=("sequence", "pathway"),
                    default="sequence",
                    help="with --hide-spectators on a branching mechanism: "
                         "'sequence' keeps any species that reacts in some "
                         "pathway, so all pathways show the same molecules "
                         "(default); 'pathway' hides per pathway, which is "
                         "less cluttered but not comparable across rows")
    ap.add_argument("--max-paths", type=int, default=8,
                    help="with --seq: most pathways to draw when the mechanism "
                         "branches (default: 8)")
    args = ap.parse_args()

    common = dict(keep_hs=args.keep_hs, show_maps=args.maps,
                  highlight=args.highlight, max_width=args.max_width,
                  hide_spectators=args.hide_spectators)

    if args.reaction:
        render_single(args.reaction, args.out, **common)

    elif args.file and args.seq:
        with open(args.file) as fh:
            lines = [l for l in fh
                     if l.strip()
                     and l.rstrip("\n").rsplit("|", 1)[-1].strip() == str(args.seq)]
        if not lines:
            sys.exit(f"no lines with sequence_idx {args.seq} in {args.file}")
        render_mechanism(lines, args.out,
                         number_steps=not args.no_step_numbers,
                         max_paths=args.max_paths,
                         spectator_scope=args.spectator_scope, **common)

    elif args.file and args.line:
        with open(args.file) as fh:
            for i, raw in enumerate(fh, start=1):
                if i == args.line:
                    render_single(raw, args.out, **common)
                    break
            else:
                sys.exit(f"{args.file} has fewer than {args.line} lines")

    else:
        ap.error("give a reaction string, or --file with --line or --seq")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"error: {e}")
