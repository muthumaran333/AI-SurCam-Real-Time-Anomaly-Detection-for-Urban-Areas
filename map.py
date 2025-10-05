from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf_report(heatmap_image_path, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    c.drawString(100, 750, "Object Detection Report")
    c.drawImage(heatmap_image_path, 100, 400, width=400, height=300)
    c.save()

# Example of creating a report
create_pdf_report('heatmap.png', 'report.pdf')
