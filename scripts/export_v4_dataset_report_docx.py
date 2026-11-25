import os
import sys
import subprocess
from pathlib import Path

try:
    import docx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-docx"])
    import docx

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_PATH = Path(r"D:\SYNEXAN\Project\Cry_Baby_Detection\reports\Bao_Cao_Thong_Ke_Va_Xu_Ly_Dataset_V4.docx")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\thiet\.gemini\antigravity-cli\brain\8572cbe7-5a3c-48d4-b5e9-c06fe9bc9ae1")

def set_cell_bg(cell, hex_color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_cell_margins(cell, top=80, bottom=80, left=110, right=110):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_styled_table(doc, headers, rows, col_widths=None, bold_last_row=False):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    # Header row
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        cell = hdr_cells[i]
        set_cell_bg(cell, "1E3A5F")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        is_last = bold_last_row and (r_idx == len(rows) - 1)
        bg = "E2E8F0" if is_last else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row_data):
            cell = row_cells[c_idx]
            set_cell_bg(cell, bg)
            set_cell_margins(cell, top=75, bottom=75, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(val)
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            if is_last:
                run.bold = True

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()
    return table

def add_heading_custom(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    if level == 1:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
    else:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x2B, 0x54, 0x7E)
    return p

def add_bullet(doc, label, value):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    r1 = p.add_run(label)
    r1.bold = True
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(11.5)
    r2 = p.add_run(value)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(11.5)
    return p

def build_docx():
    doc = Document()

    # Page setup: A4, standard margins
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Default Normal style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11.5)

    # Title block
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(4)
    r_title = title_p.add_run("BÁO CÁO THỐNG KÊ THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU\nBỘ DỮ LIỆU VISION V4 CLEAN-BALANCED (ADULTS VS KIDS)")
    r_title.bold = True
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Phiên bản huấn luyện: YOLO11-M (thittn/train-yolo11-m-v4-clean-balanced)")
    r_sub.italic = True
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Section 1
    add_heading_custom(doc, "1. Thông tin chung & Đường dẫn lưu trữ", level=1)
    add_bullet(doc, "Dataset Kaggle V4 đã xử lý: ", "thittn/vision-v4-clean-balanced (https://www.kaggle.com/datasets/thittn/vision-v4-clean-balanced) — 307.88 MB, định dạng YOLO.")
    add_bullet(doc, "Notebook huấn luyện Kaggle: ", "https://www.kaggle.com/code/thittn/train-yolo11-m-v4-clean-balanced")
    add_bullet(doc, "Thư mục lưu trữ cục bộ: ", r"D:\SYNEXAN\Project\Cry_Baby_Detection\data\processed\vision_v4_clean_balanced")
    add_bullet(doc, "Script xây dựng & làm sạch dataset: ", "scripts/build_v4_clean_balanced_dataset.py")
    add_bullet(doc, "Hệ nhãn thống nhất (2 lớp): ", "0: adults (người lớn) | 1: kids (trẻ em & trẻ sơ sinh).")

    # Section 2
    add_heading_custom(doc, "2. Nguồn dữ liệu gốc & Quy tắc ánh xạ nhãn", level=1)
    p_intro2 = doc.add_paragraph(
        "Bộ dữ liệu V4 Clean-Balanced được tổng hợp từ 3 nguồn dữ liệu công khai có độ sạch cao trên Kaggle và Roboflow Universe, "
        "sau đó được ánh xạ đồng nhất về 2 lớp mục tiêu (adults và kids) và trải qua quy trình kiểm toán, lọc nhiễu tự động."
    )
    p_intro2.paragraph_format.line_spacing = 1.2

    add_styled_table(
        doc,
        headers=["STT", "Tên nguồn", "Đường dẫn gốc", "Số ảnh gốc", "Số ảnh giữ lại (V4)", "Quy tắc gộp nhãn về 2 lớp"],
        rows=[
            [
                "1",
                "Kids and Adults Detection (kaushi_)",
                "https://www.kaggle.com/datasets/kaushigihanml/kids-and-adults-detection",
                "505",
                "505 (100%)\n(+354 bản nhân đôi chỉ ở tập Train)",
                "Adult -> 0 (adults)\nKid -> 1 (kids)",
            ],
            [
                "2",
                "Child Adult Detection (anir_)",
                "https://universe.roboflow.com/aniruddha-jmp5a/child-adult-detection-u81pm",
                "1,484",
                "1,389 (93.6%)\n(loại 95 ảnh lỗi / sót nhãn)",
                "therapists -> 0 (adults)\nchild -> 1 (kids)",
            ],
            [
                "3",
                "True Baby (tb_)",
                "https://universe.roboflow.com/baby-7psdb/true-baby",
                "1,253",
                "440 (35.1%)\n(rút gọn & lọc nhiễu)",
                "adult -> 0 (adults)\nbaby, child -> 1 (kids)",
            ],
            [
                "",
                "TỔNG CỘNG",
                "3 nguồn hợp nhất (Zero Leakage)",
                "3,242 ảnh gốc",
                "2,334 ảnh gốc duy nhất\n(2,688 ảnh tính cả 354 bản kaushi x2 ở Train)",
                "2,149 box adults\n3,752 box kids",
            ],
        ],
        col_widths=[0.45, 1.25, 1.75, 0.75, 1.3, 1.1],
        bold_last_row=True,
    )

    # Section 3
    add_heading_custom(doc, "3. Tỷ lệ chia tập & Thống kê chi tiết từng tập (70 / 15 / 15 — Zero Data Leakage)", level=1)
    p_intro3 = doc.add_paragraph(
        "Dữ liệu được chia phân tầng (Stratified Split) theo từng tổ hợp (nguồn, có_adults, có_kids) theo đúng tỷ lệ "
        "70% Train / 15% Valid / 15% Test trên 2,334 ảnh gốc duy nhất, đảm bảo cam kết 0% rò rỉ dữ liệu (Zero Data Leakage) "
        "giữa các tập. Chỉ sau khi chia tách hoàn tất, hệ thống mới thực hiện nhân đôi (oversampling x2) đối với các ảnh kaushi "
        "duy nhất trong tập Train; tập Valid và tập Test được giữ nguyên 100%."
    )
    p_intro3.paragraph_format.line_spacing = 1.2

    add_styled_table(
        doc,
        headers=["Tập dữ liệu", "Tỷ lệ gốc", "Tổng số ảnh", "Chi tiết ảnh theo nguồn", "Box adults (0)", "Box kids (1)", "Tỷ lệ kids:adults"],
        rows=[
            [
                "Train",
                "70%\n(+kaushi x2)",
                "1,987\n(1,633 gốc + 354 dup)",
                "- kaushi: 354 gốc + 354 nhân đôi (= 708, chiếm 35.6%)\n- aniruddha: 971\n- true_baby: 308",
                "1,564\n(kaushi: 516,\nanir: 1,012,\ntb: 36)",
                "2,773\n(kaushi: 1,282,\nanir: 1,158,\ntb: 333)",
                "1.77 : 1",
            ],
            [
                "Valid",
                "15%",
                "350",
                "- kaushi: 76\n- aniruddha: 208\n- true_baby: 66",
                "288\n(kaushi: 52,\nanir: 230,\ntb: 6)",
                "510\n(kaushi: 152,\nanir: 286,\ntb: 72)",
                "1.77 : 1",
            ],
            [
                "Test",
                "15%",
                "351",
                "- kaushi: 75\n- aniruddha: 210\n- true_baby: 66",
                "297\n(kaushi: 55,\nanir: 236,\ntb: 6)",
                "469\n(kaushi: 138,\nanir: 259,\ntb: 72)",
                "1.58 : 1",
            ],
            [
                "TỔNG",
                "100%",
                "2,688",
                "2,334 ảnh gốc duy nhất + 354 bản nhân đôi kaushi (chỉ ở Train)",
                "2,149",
                "3,752",
                "1.75 : 1",
            ],
        ],
        col_widths=[0.65, 0.7, 0.85, 1.85, 0.9, 0.9, 0.75],
        bold_last_row=True,
    )

    # Section 4
    add_heading_custom(doc, "4. Quy trình tiền xử lý, làm sạch dữ liệu & Lý do kỹ thuật", level=1)
    add_styled_table(
        doc,
        headers=["Bước xử lý", "Cách thực hiện cụ thể", "Kết quả lọc", "Lý do kỹ thuật"],
        rows=[
            [
                "1. Gộp nhãn chuẩn hóa",
                "Ánh xạ therapists / Adult / adult -> 0 (adults) và Kid / child / baby -> 1 (kids).",
                "Đồng nhất 2 lớp sạch trên cả 3 nguồn.",
                "Loại bỏ sai khác tên lớp giữa Roboflow và Kaggle (đặc biệt lớp therapists ở aniruddha thực chất là người lớn / giáo viên).",
            ],
            [
                "2. Lọc box rác li ti & quá khổ",
                "Xóa bounding box có diện tích < 0.8% ảnh (hoặc cạnh < 3%) và box lỗi > 88% ảnh; loại bỏ ảnh rỗng sau khi xóa.",
                "Xóa box nhiễu trên 68 ảnh; loại 36 ảnh rỗng / chỉ chứa box rác.",
                "Tránh mô hình học các chấm người xa 10–15px ngoài đường chân trời không phân biệt được tuổi hoặc các box lỗi phủ kín khung hình.",
            ],
            [
                "3. Tự động lọc ảnh bỏ sót nhãn người lớn (Auto-Audit)",
                "Dùng yolo11n.pt (COCO lớp person, conf >= 0.65, area >= 8%) quét aniruddha & true_baby; nếu có người lớn cận cảnh mà IoU < 0.15 với mọi Ground Truth thì loại bỏ ảnh.",
                "Loại 65 ảnh độc hại (62 từ aniruddha, 3 từ true_baby).",
                "Nếu ảnh có người lớn ngay tiền cảnh nhưng không có nhãn, YOLO sẽ bị phạt oan khi dự đoán adults, làm tụt Recall và tăng nhầm lẫn adults <-> kids.",
            ],
            [
                "4. Lọc nhiễu muối tiêu & Tái cân bằng true_baby",
                "Tính điểm nhiễu xung (độ lệch so với medianBlur 3x3 > 45), loại ảnh nhiễu nặng (> 1.5%); giữ 100% ảnh true_baby có adults (40), chọn lọc 200 ảnh CCTV xám sạch nhất + 200 ảnh màu sạch nhất.",
                "Rút gọn true_baby từ 1,253 -> 440 ảnh sạch nhất.",
                "true_baby gốc có tỷ lệ mất cân bằng cực đoan (27.7 : 1 kids:adults, 96.8% chỉ có 1 bé) và nhiều ảnh rắc nhiễu muối tiêu từ Roboflow. Rút gọn còn 440 ảnh vừa đủ giữ khả năng nhận trẻ sơ sinh CCTV đêm vừa kéo tỷ lệ chung về 1.75 : 1.",
            ],
            [
                "5. Chia 70/15/15 Zero Leakage + Nhân đôi kaushi x2 trong Train",
                "Chia phân tầng 70/15/15 trước, sau đó chỉ nhân đôi kaushi (354 -> 708 ảnh) trong tập train; valid và test giữ nguyên 100%.",
                "Tăng tỷ trọng ảnh HD gốc kaushi trong Train từ ~15% lên 35.6%.",
                "kaushi là nguồn duy nhất giữ nguyên độ phân giải HD (2048x1365), không bị Roboflow ép vuông và có tỷ lệ đồng xuất hiện cả người lớn + trẻ em cao nhất (68.7%). Nhân đôi trong Train giúp tăng trọng số học cho ảnh sắc nét mà không gây rò rỉ dữ liệu sang valid/test.",
            ],
        ],
        col_widths=[1.2, 1.85, 1.25, 2.3],
    )

    # Section 5: Visual Audit Evidence
    add_heading_custom(doc, "5. Minh họa trực quan kết quả kiểm toán & làm sạch dữ liệu", level=1)
    visuals = [
        ("audit_v4_dropped_missing_labels.jpg", "Hình 1: Các ảnh độc hại bị loại bỏ do bỏ sót nhãn người lớn cận cảnh (khung đỏ MISSING PERSON)."),
        ("audit_v4_tiny_boxes_removed.jpg", "Hình 2: Các ảnh được giữ lại sau khi xóa bỏ box nhiễu li ti (< 0.8%) và box lỗi quá khổ (> 88%) (khung đỏ REMOVED NOISE)."),
        ("audit_v4_clean_kept_samples.jpg", "Hình 3: Mẫu ảnh sạch được giữ lại trong tập Train V4 Clean-Balanced từ 3 nguồn (Hàng 1: kaushi HD, Hàng 2: aniruddha, Hàng 3: true_baby)."),
    ]
    for fname, caption in visuals:
        fpath = ART_DIR / fname
        if fpath.exists():
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(str(fpath), width=Inches(6.4))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            rc = p_cap.add_run(caption)
            rc.italic = True
            rc.font.name = "Times New Roman"
            rc.font.size = Pt(10)
            rc.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    doc.save(str(OUT_PATH))
    print(f"SUCCESS: Saved Word report to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    build_docx()
