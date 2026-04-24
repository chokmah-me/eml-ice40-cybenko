# Zenodo + GitHub release instructions

## 0. What goes where

**One deposit, two artifacts tied by DOI.** Zenodo has two ways to create
a record. Use the GitHub integration path: it auto-archives your repo on
every tagged release and mints a versioned DOI. The paper PDF can either
ride along inside the repo (simplest) or be a separate "Publication"
record linked by DOI (cleaner for citations). I recommend **Option A**
below for a first submission. Option B is for later.

## Option A (recommended): single combined Zenodo record via GitHub release

The entire GitHub repo (code + CSV + paper PDF + figure) becomes one
Zenodo record with one DOI. Cleaner, fewer moving parts, one thing to
cite.

### Step 1. Assemble the GitHub repo locally

In your local clone of `chokmah-me/eml-ice40-cybenko`, put exactly these
files at the repo root:

```
README.md                        <- provided in this bundle
LICENSE                          <- provided (MIT)
CITATION.cff                     <- provided
.zenodo.json                     <- provided
.gitignore                       <- provided
PAPER1_DRAFT_v2.md               <- from project (or use reflowed version)
PAPER1_DRAFT_v2.pdf              <- your exported PDF
figure1_heatmap_v2.png           <- from project
snapping_v2_final.csv            <- from project
eml_layer_v2.py                  <- YOU MUST ADD THIS from your local machine
experiment_v2.py                 <- YOU MUST ADD THIS from your local machine
```

The two `.py` files are referenced in the paper's "Data and code" section
but are not in the project directory I have access to. Grab them from
your local machine.

Before committing: open `CITATION.cff` and `.zenodo.json` and verify the
author name, email, and affiliation. Edit if needed.

### Step 2. Commit and push to GitHub

```pwsh
cd <path-to-your-local-clone>
git add README.md LICENSE CITATION.cff .zenodo.json .gitignore `
        PAPER1_DRAFT_v2.md PAPER1_DRAFT_v2.pdf figure1_heatmap_v2.png `
        snapping_v2_final.csv eml_layer_v2.py experiment_v2.py
git commit -m "Initial release: code, data, and paper v2.0"
git push origin main
```

Confirm on github.com that the repo is **public**. Zenodo cannot archive
private repos.

### Step 3. Enable Zenodo-GitHub integration (one-time setup)

1. Go to https://zenodo.org and "Log in with GitHub".
2. Authorize Zenodo to access your GitHub account.
3. Go to https://zenodo.org/account/settings/github/.
4. Find `chokmah-me/eml-ice40-cybenko` in the list and flip its toggle to
   **ON**.

Critical: the toggle must be ON *before* you create the release. Zenodo
only archives releases made after the toggle is flipped. If you already
made a release, you'll need to create a new one (see Step 4).

### Step 4. Create the GitHub release

On github.com, go to your repo → Releases → "Draft a new release".

- Tag version: `v1.0.0`
- Release title: `v1.0.0 - paper and code release`
- Description: one short paragraph. Example:

  > First public release accompanying the paper "Valid and False Snapping
  > in EML Expression Trees: The Basin Selection Problem" (Bilar, Chokmah
  > LLC, April 2026). Includes the EML layer and training driver, the
  > full 240-run results CSV with pre- and post-snap losses, Figure 1,
  > and the paper PDF.

Click **Publish release**.

### Step 5. Wait for Zenodo to ingest

Usually 1-5 minutes. Check https://zenodo.org/account/settings/github/,
click on your repo, and you'll see the new release with its DOI.

You'll get two DOIs:
- a **concept DOI** that always resolves to the latest version
- a **version DOI** specific to `v1.0.0`

Put the concept DOI badge in your README. Zenodo shows the markdown
snippet on the repo's integration page. Commit and push that update.

### Step 6. Verify

Click the DOI. Confirm:
- title is correct
- author is correct
- license shows MIT
- the repo ZIP is attached
- description makes sense
- keywords are populated

If any metadata is wrong, you can edit it directly on the Zenodo record
page ("Edit" button). Metadata edits don't require a new release. Code
or data changes do.

## Option B (later, if you want a separate paper record)

Publish the paper PDF as its own Zenodo "Publication" record:
1. zenodo.org → "New upload".
2. Upload type: Publication → Preprint (or Journal article if accepted).
3. Upload `PAPER1_DRAFT_v2.pdf`.
4. Fill metadata (title, author, abstract, license CC BY 4.0).
5. Under "Related/alternate identifiers", add the GitHub-Zenodo software
   DOI from Option A with relation "compiles" or "is supplemented by".
6. Publish.

You now have two DOIs cross-linked: one for the paper, one for the code.
Cite both.

## Updating later

For a v1.0.1 bugfix or v2.0.0 expansion:
1. Commit changes to the repo.
2. Update the version in `CITATION.cff` and `.zenodo.json`.
3. Create a new GitHub release (e.g. `v1.0.1`).
4. Zenodo mints a new version DOI automatically under the same concept
   DOI.

## Pre-flight checklist

- [ ] Repo is public
- [ ] `README.md` renders correctly on github.com
- [ ] `LICENSE` file present (MIT)
- [ ] `CITATION.cff` has your correct name and affiliation
- [ ] `.zenodo.json` has correct author and license
- [ ] `eml_layer_v2.py` and `experiment_v2.py` are committed
- [ ] `snapping_v2_final.csv` is committed
- [ ] `PAPER1_DRAFT_v2.pdf` is committed (your final export)
- [ ] Zenodo-GitHub integration enabled for this repo
- [ ] Tag `v1.0.0` does not yet exist
