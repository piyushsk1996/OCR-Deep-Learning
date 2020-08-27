from pdf2image import convert_from_path

pages = convert_from_path("PDF's/test.pdf", 500)

for count, page in enumerate(pages):
    print(count)
    page.save('Images_from_pdf/' + str(count) + '.jpg', 'JPEG')



