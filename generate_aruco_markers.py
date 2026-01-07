"""
Generate ArUco markers for backdrop calibration (A4 size, accurate physical dimensions).

Creates 4 ArUco markers (IDs 0-3) on TWO A4 pages with precise sizing for accurate printing.
2 markers per page allows for larger marker sizes (12cm recommended for better accuracy).
"""

import cv2
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from PIL import Image

# PDF unit system: 1 point = 1/72 inch
# We calculate all dimensions manually to ensure accuracy

def cm_to_points(cm_value):
    """Convert centimeters to PDF points (1 inch = 72 points, 1 inch = 2.54 cm)."""
    return cm_value * (72.0 / 2.54)

def mm_to_points(mm_value):
    """Convert millimeters to PDF points."""
    return mm_value * (72.0 / 25.4)

# A4 page dimensions in points (210mm × 297mm) - DO NOT SCALE
A4_WIDTH_POINTS = mm_to_points(210)   # 595.2755905511812 points
A4_HEIGHT_POINTS = mm_to_points(297)  # 841.8897637795277 points
A4 = (A4_WIDTH_POINTS, A4_HEIGHT_POINTS)

# Empirically determined correction factor for marker sizes
# Markers were printing at ~87.87% of specified size, so we scale by 1/0.8787
# This compensates for PDF rendering differences
MARKER_SCALE_CORRECTION = 1.138


def generate_aruco_marker(marker_id, marker_size_px=500, dictionary=cv2.aruco.DICT_4X4_50):
    """
    Generate a single ArUco marker image.

    Args:
        marker_id: ID number for the marker (0-49 for DICT_4X4_50)
        marker_size_px: Size of the generated image in pixels
        dictionary: ArUco dictionary type

    Returns:
        Numpy array containing the marker image
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
    return marker_img


def create_a4_marker_sheet(output_path="aruco_markers_for_backdrop.pdf",
                           marker_size_cm=12.0):
    """
    Generate a PDF with 4 ArUco markers for backdrop calibration.

    Layout automatically adjusts based on marker size:
    - Markers ≤ 11cm: 2 per page (2 pages total)
    - Markers > 11cm: 1 per page (4 pages total)
    - Maximum marker size: ~20cm (limited by A4 width)

    Space-efficient layout with compact legend at bottom right corner.
    Margins automatically reduce for larger markers to maximize space.

    Args:
        output_path: Path to save the PDF file
        marker_size_cm: Physical size of each marker in cm (default 12.0, max ~20.0)
    """
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # A4 dimensions: 210mm × 297mm
    page_width, page_height = A4  # In points (1 point = 1/72 inch)

    # Convert marker size to PDF points with correction factor
    # This ensures the printed marker matches the specified physical size
    marker_size = cm_to_points(marker_size_cm) * MARKER_SCALE_CORRECTION

    # Adjust margins based on marker size - use minimal margins for large markers
    if marker_size_cm >= 15.0:
        safe_margin = mm_to_points(2)  # Minimal margin for very large markers (15-20cm)
    elif marker_size_cm >= 12.0:
        safe_margin = mm_to_points(3)  # Reduced margin for large markers (12-15cm)
    else:
        safe_margin = mm_to_points(5)  # Standard margin for smaller markers

    # Calculate printable area
    printable_width = page_width - 2 * safe_margin

    # Check if marker fits in printable area (legend is at bottom, only needs 15mm on right)
    if marker_size > printable_width - mm_to_points(15):  # Account for compact legend at bottom
        raise ValueError(f"Marker size {marker_size_cm}cm is too large for A4 page (max ~20cm)")

    # Automatically choose layout based on marker size
    if marker_size_cm > 11.0:
        # Large markers: 1 per page (4 pages total)
        pages = [
            [(0, "TL")],      # Page 1: Top-Left
            [(1, "TR")],      # Page 2: Top-Right
            [(2, "BL")],      # Page 3: Bottom-Left
            [(3, "BR")]       # Page 4: Bottom-Right
        ]
    else:
        # Standard markers: 2 per page (2 pages total)
        pages = [
            [(0, "TL"), (1, "TR")],      # Page 1: Top-Left, Top-Right
            [(2, "BL"), (3, "BR")]       # Page 2: Bottom-Left, Bottom-Right
        ]

    # Create PDF canvas
    c = canvas.Canvas(output_path, pagesize=A4)

    total_pages = len(pages)

    for page_num, markers in enumerate(pages, 1):
        if page_num > 1:
            c.showPage()  # Start new page

        # Minimal page number in top right corner
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.gray)
        c.drawRightString(page_width - safe_margin, page_height - safe_margin - mm_to_points(3),
                         f"{page_num}/{total_pages}")
        c.setFillColor(colors.black)

        # Layout calculation based on number of markers per page
        num_markers_this_page = len(markers)
        v_spacing = mm_to_points(20) if num_markers_this_page > 1 else 0  # Spacing only if multiple markers

        # Calculate vertical positions to maximize space
        if num_markers_this_page == 1:
            # Single marker: center vertically
            start_y = (page_height - marker_size) / 2
        else:
            # Multiple markers: distribute vertically
            total_markers_height = num_markers_this_page * marker_size + v_spacing
            start_y = (page_height - total_markers_height) / 2

        # Position markers on left side, leave room for legend on right
        marker_x = safe_margin + mm_to_points(10)

        # Track marker positions for legend placement
        marker_positions = []

        # Generate and place each marker
        for idx, (marker_id, label) in enumerate(markers):
            # Calculate position (top to bottom)
            # Use (num_markers - 1 - idx) to correctly handle both single and multiple markers
            y = start_y + (num_markers_this_page - 1 - idx) * (marker_size + v_spacing)

            # Track this marker's position
            marker_positions.append({
                'x': marker_x,
                'y': y,
                'width': marker_size,
                'height': marker_size
            })

            # Generate marker with high resolution
            marker_img = generate_aruco_marker(marker_id, marker_size_px=1200)

            # Convert to PIL Image
            pil_img = Image.fromarray(marker_img)

            # Use ImageReader to wrap the PIL Image for ReportLab
            img_reader = ImageReader(pil_img)

            # Draw marker on PDF at exact size
            c.drawImage(img_reader, marker_x, y, width=marker_size, height=marker_size, mask='auto')

            # Draw border around marker
            c.setStrokeColor(colors.black)
            c.setLineWidth(2.0)
            c.rect(marker_x, y, marker_size, marker_size, stroke=1, fill=0)

            # Add small ID label next to marker (compact)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(marker_x + marker_size + mm_to_points(3), y + marker_size/2, f"ID {marker_id}")

        # Add rotated legend with dynamic positioning to avoid marker overlap
        add_compact_legend(c, page_width, page_height, safe_margin, marker_positions)

    # Save the PDF
    c.save()

    # Determine layout description
    markers_per_page_text = "1 per page" if total_pages == 4 else "2 per page"

    print(f"\n{'='*70}")
    print("✓ ARUCO MARKERS GENERATED - READY TO PRINT")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Marker Size: {marker_size_cm}cm × {marker_size_cm}cm")
    print(f"Format: A4, {total_pages} pages, print-ready ({markers_per_page_text})")
    print()
    print(f"{'='*70}")
    print("PRINTING INSTRUCTIONS")
    print(f"{'='*70}")
    print("1. Open PDF and print with DEFAULT settings")
    print("2. Page size will auto-match A4 (210mm × 297mm)")
    print("3. ✓ VERIFY: Use a ruler to measure printed markers")
    print(f"   Must be EXACTLY {marker_size_cm}cm × {marker_size_cm}cm")
    print("4. Cut markers along black borders")
    print()
    print("TROUBLESHOOTING:")
    print(f"• If markers ≠ {marker_size_cm}cm: Select 'Actual size'/'100% scale'")
    print("• DO NOT use 'Fit to page' or 'Shrink to fit'")
    print()
    print(f"{'='*70}")
    print("LAYOUT")
    print(f"{'='*70}")
    print(f"• Markers: Centered ({markers_per_page_text})")
    print("• Legend: Bottom right corner (compact, rotated 90°)")
    print(f"• Margins: {safe_margin / mm_to_points(1):.1f}mm (optimized for {marker_size_cm}cm markers)")
    print("• Page #: Top right corner")
    print()
    print("BACKDROP PLACEMENT:")
    print("  ID 0 → Top-Left    |  ID 1 → Top-Right")
    print("  ID 2 → Bottom-Left |  ID 3 → Bottom-Right")
    print()
    print("✓ Markers flat, no wrinkles")
    print(f"✓ Size verified = {marker_size_cm}cm")
    print("✓ All 4 markers visible in photos")
    print(f"{'='*70}\n")


def add_vertical_ruler(c, page_width, page_height, marker_size_cm, safe_margin):
    """Add a vertical verification ruler on the right side of the page."""
    # Position ruler further to the right
    ruler_x = page_width - safe_margin - mm_to_points(10)

    # Draw vertical ruler line from top margin to just above legend
    # Start from just below top margin
    ruler_top_y = page_height - safe_margin - mm_to_points(15)
    # End just above the legend (legend is 40mm tall + 5mm margin + 5mm gap = 50mm from bottom)
    ruler_bottom_y = safe_margin + mm_to_points(50)

    # Draw vertical ruler line
    c.setStrokeColor(colors.red)
    c.setLineWidth(3)
    c.line(ruler_x, ruler_bottom_y, ruler_x, ruler_top_y)

    # Draw tick marks at regular intervals matching the marker size
    c.setLineWidth(2)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.red)

    num_cm_marks = int(marker_size_cm)

    # Draw tick marks and labels
    for i in range(num_cm_marks + 1):
        tick_y = ruler_bottom_y + cm_to_points(i)
        if tick_y <= ruler_top_y:
            # Draw tick mark
            c.line(ruler_x - mm_to_points(3), tick_y, ruler_x + mm_to_points(3), tick_y)
            # Label every tick
            c.drawCentredString(ruler_x + mm_to_points(7), tick_y - mm_to_points(1.5), str(i))

    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)


def add_compact_legend(c, page_width, page_height, safe_margin, marker_positions):
    """Add a compact legend at the absolute bottom right corner of the page.

    Positioned at the very bottom right to avoid interfering with large markers.

    Args:
        c: ReportLab canvas
        page_width: Page width in points
        page_height: Page height in points
        safe_margin: Safe margin in points (not used, kept for compatibility)
        marker_positions: Marker positions (not used, kept for compatibility)
    """

    # Legend dimensions (horizontal, will be rotated)
    legend_width_horizontal = mm_to_points(40)  # Width when horizontal
    legend_height_horizontal = mm_to_points(15)  # Height when horizontal

    # Position at absolute bottom right corner with minimal margins
    # When rotated 90° CCW, the legend will extend upward from bottom right
    margin_from_edge = mm_to_points(2)  # Minimal margin from page edge

    legend_x = page_width - margin_from_edge
    legend_y = margin_from_edge

    # Save state for rotation
    c.saveState()

    # Translate to the legend position and rotate 90° counter-clockwise
    c.translate(legend_x, legend_y)
    c.rotate(90)

    # Now draw the legend in the rotated coordinate system
    # Draw legend box (after rotation: width becomes height, height becomes width)
    c.setStrokeColor(colors.gray)
    c.setLineWidth(0.5)
    c.rect(0, 0, legend_width_horizontal, legend_height_horizontal, stroke=1, fill=0)

    # Legend title - compact to fit smaller box
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.black)
    c.drawString(mm_to_points(2), mm_to_points(11), "Placement:")

    # Legend items - compact format
    c.setFont("Helvetica", 6)
    legend_items = [
        "0:TL  1:TR",
        "2:BL  3:BR"
    ]

    y_pos = mm_to_points(7)
    for item in legend_items:
        c.drawString(mm_to_points(2), y_pos, item)
        y_pos -= mm_to_points(3.5)

    # Restore state
    c.restoreState()

    c.setStrokeColor(colors.black)
    c.setFillColor(colors.black)


def save_individual_marker_pngs(output_dir="aruco_markers", marker_ids=[0, 1, 2, 3]):
    """
    Save individual marker images as high-resolution PNG files.

    Args:
        output_dir: Directory to save marker images
        marker_ids: List of marker IDs to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for marker_id in marker_ids:
        # Generate high-res marker
        marker_img = generate_aruco_marker(marker_id, marker_size_px=2000)

        # Save as PNG
        filename = output_path / f"aruco_marker_{marker_id}.png"
        cv2.imwrite(str(filename), marker_img)
        print(f"Saved: {filename}")

    print(f"\nAll {len(marker_ids)} markers saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ArUco markers for backdrop calibration (2 markers per page)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aruco_markers_for_backdrop.pdf",
        help="Output PDF path (default: aruco_markers_for_backdrop.pdf)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=12.0,
        help="Marker size in cm (default: 12.0 for better accuracy)"
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also save individual high-res PNG files"
    )

    args = parser.parse_args()

    # Generate PDF with markers
    create_a4_marker_sheet(args.output, args.size)

    # Optionally save PNG files
    if args.png:
        save_individual_marker_pngs()
