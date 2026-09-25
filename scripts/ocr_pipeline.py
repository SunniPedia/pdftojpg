"""
Gausia Tarbiyati Nesab - পেইজ ইমেজ (501.jpg, 502.jpg ...) থেকে বাংলা-আরবি টেক্সট বের করে
একটা সিঙ্গেল আউটপুট ফাইলে লেখে।

সীমাবদ্ধতা (স্বীকার করেই রাখা হলো, কোড এগুলো "সমাধান" করে না):
  1. বাংলা আর আরবি আলাদা script-family, তাই EasyOCR-এ একসাথে লোড করা যায় না।
     তাই প্রতি পেইজে দুইবার scan করা হয় (bn+en, এবং ar) এবং bounding-box
     পজিশন দিয়ে লাইন সাজিয়ে merge করা হয়। Overlap হলে confidence বেশি যেটার
     সেটা রাখা হয় - এটা perfect না, বিশেষ করে এক লাইনে বাংলা+আরবি মেশানো থাকলে।
  2. HEADING ডিটেকশন raw text থেকে সম্ভব না (font/bold info লাগে যেটা EasyOCR
     দেয় না নির্ভরযোগ্যভাবে) - তাই heading রুল (rule 2) এখানে প্রয়োগ করা হয়নি,
     manual review দরকার।
  3. আরবি হরকত/তাশদীদ recognition accuracy printed harakat-সহ টেক্সটেও কম
     হতে পারে - output ম্যানুয়ালি যাচাই করা জরুরি, বিশেষ করে اللَّه জাতীয় শব্দে।
  4. আয়াত মার্কার ۝ প্রায়ই ভুল করে অন্য গ্লিফ হিসেবে পড়া হয় (rule 7 এর সতর্কতা
     এখানেই কেন) - এই স্ক্রিপ্ট শুধু existing ۝ চিহ্ন প্রিজার্ভ করে, ভুল OCR
     এর কারণে হারিয়ে যাওয়া ۝ ফিরিয়ে দিতে পারে না।

যা এই স্ক্রিপ্ট মেকানিক্যালি করে (rule 1, 3, 5, 6-partial, 7-partial):
  - প্রথম লাইন হেডার-সদৃশ হলে বাদ দেয় (rule 1)
  - ।<digit> প্যাটার্ন পেলে ফুটনোট মার্কার হিসেবে চিহ্নিত করে এবং
    [FOOTNOTE ...] ব্লক আলাদা করে (rule 3) - তবে ফুটনোটের পূর্ণ টেক্সট কোথায়
    শেষ হচ্ছে তা bold heuristic ছাড়া অনুমান, তাই ভুল হতে পারে
  - শব্দ-শেষ ঃ কে : তে বদলায়, শব্দ-মাঝে অপরিবর্তিত রাখে (rule 5)
  - তাশদীদ ও হরকত ক্যারেক্টার ড্রপ হয়েছে কিনা এমন সন্দেহজনক জায়গা flag করে (rule 6)
  - ۝ + সংখ্যা প্যাটার্ন অক্ষত আছে কিনা যাচাই করে, না থাকলে flag করে (rule 7)
"""

import argparse
import re
import sys
from pathlib import Path

import easyocr

FOOTNOTE_MARKER_RE = re.compile(r"।([০-৯]+)")
WORD_FINAL_BISARGA_RE = re.compile(r"ঃ(?=\s|$|[।,.!?])")
AYAH_MARK_RE = re.compile(r"۝\s*[٠-٩]+")
HEADER_HINT_RE = re.compile(r"গাউসিয়া তারবিয়াতী নেসাব|[০-৯]{1,4}\s*\.{3,}")


def find_page_image(img_dir: Path, page: int, width_hint: int) -> Path | None:
    # আগে leading-zero সহ চেষ্টা (যেমন 0535.jpg, ইনপুট যত ডিজিটের ছিল সেই width),
    # না পেলে plain নম্বর (535.jpg) দিয়ে চেষ্টা
    candidates = [str(page).zfill(width_hint), str(page)]
    for name in candidates:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidate = img_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
    return None


def ocr_page(image_path: Path, reader_bn, reader_ar):
    """দুই reader দিয়ে scan করে, bbox-এর top-y অনুযায়ী sort করে merged line list দেয়।"""
    boxes = []
    for reader, tag in ((reader_bn, "bn"), (reader_ar, "ar")):
        for bbox, text, conf in reader.readtext(str(image_path)):
            if not text.strip():
                continue
            top_y = min(p[1] for p in bbox)
            left_x = min(p[0] for p in bbox)
            boxes.append({"y": top_y, "x": left_x, "text": text, "conf": conf, "tag": tag})

    # একই এলাকায় (y কাছাকাছি) দুই reader থেকে detection এলে confidence বেশিটা রাখা
    boxes.sort(key=lambda b: (round(b["y"] / 15), b["x"]))
    merged = []
    for b in boxes:
        if merged:
            last = merged[-1]
            same_line = abs(b["y"] - last["y"]) < 15
            overlap_x = abs(b["x"] - last["x"]) < 20
            if same_line and overlap_x:
                if b["conf"] > last["conf"]:
                    merged[-1] = b
                continue
        merged.append(b)

    # লাইন অনুযায়ী গ্রুপ করে বাম-থেকে-ডান জোড়া লাগানো
    lines = []
    current_y = None
    current_line = []
    for b in merged:
        if current_y is None or abs(b["y"] - current_y) > 15:
            if current_line:
                lines.append(" ".join(w["text"] for w in sorted(current_line, key=lambda w: w["x"])))
            current_line = [b]
            current_y = b["y"]
        else:
            current_line.append(b)
    if current_line:
        lines.append(" ".join(w["text"] for w in sorted(current_line, key=lambda w: w["x"])))

    return lines


def apply_rules(lines: list[str], page: int, warnings: list[str]) -> str:
    if lines and HEADER_HINT_RE.search(lines[0]):
        lines = lines[1:]

    out_blocks = []
    footnotes = []
    for line in lines:
        # rule 5: word-final ঃ -> :
        line = WORD_FINAL_BISARGA_RE.sub(":", line)

        # rule 3: footnote marker
        m = FOOTNOTE_MARKER_RE.search(line)
        if m:
            marker = m.group(1)
            footnotes.append(marker)
            line = line  # মূল লাইনে মার্কার সহ রাখা হলো; [FOOTNOTE] ব্লক নিচে যোগ হবে
            out_blocks.append(line)
            out_blocks.append(f"[FOOTNOTE {marker}: << manual: এখানে ফুটনোটের পূর্ণ টেক্সট বসান >>]")
            continue

        out_blocks.append(line)

        # rule 6: তাশদীদ/হরকত সন্দেহজনক ড্রপ চেক (اللَّه-জাতীয় শব্দে শাদ্দা না থাকলে flag)
        if "الله" in line and "لَّه" not in line and "للّه" not in line:
            warnings.append(f"[page {page}] সম্ভাব্য তাশদীদ মিসিং: {line[:40]}...")

        # rule 7: ۝ চিহ্ন থাকলে ফরম্যাট ঠিক আছে কিনা যাচাই
        if "۝" in line and not AYAH_MARK_RE.search(line):
            warnings.append(f"[page {page}] ۝ চিহ্নের সাথে সংখ্যা মিসিং হতে পারে: {line[:40]}...")

    return "\n".join(out_blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--page-from", required=True)  # string হিসেবে নেওয়া হলো, leading zero রক্ষা করতে
    ap.add_argument("--page-to", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        print(f"ত্রুটি: img ফোল্ডার পাওয়া যায়নি: {img_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width_hint = len(args.page_from)  # ইনপুটে যত ডিজিট (যেমন "0535" -> 4), সেই অনুযায়ী zero-pad
    page_from = int(args.page_from)
    page_to = int(args.page_to)

    print("EasyOCR মডেল লোড হচ্ছে (bn+en, ar)...", file=sys.stderr)
    reader_bn = easyocr.Reader(["bn", "en"], gpu=False)
    reader_ar = easyocr.Reader(["ar", "en"], gpu=False)

    all_pages = []
    warnings = []

    for page in range(page_from, page_to + 1):
        image_path = find_page_image(img_dir, page, width_hint)
        if image_path is None:
            warnings.append(f"[page {page}] ছবি পাওয়া যায়নি, স্কিপ করা হলো")
            continue

        print(f"OCR চলছে: {image_path.name}", file=sys.stderr)
        lines = ocr_page(image_path, reader_bn, reader_ar)
        page_text = apply_rules(lines, page, warnings)
        all_pages.append(f"--- PAGE {page} ---\n{page_text}")

    out_path.write_text("\n\n".join(all_pages), encoding="utf-8")

    if warnings:
        warn_path = out_path.with_suffix(".warnings.txt")
        warn_path.write_text("\n".join(warnings), encoding="utf-8")
        print(f"\n{len(warnings)}টা সতর্কতা - manual review দরকার। দেখুন: {warn_path}", file=sys.stderr)

    print(f"\nসম্পন্ন: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
