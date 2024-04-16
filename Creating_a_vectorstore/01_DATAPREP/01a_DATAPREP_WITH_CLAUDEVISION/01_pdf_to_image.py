from pdf2image import convert_from_path
import os
from tqdm import tqdm

def pdf_to_images(pdf_path, output_directory, dpi):
    """ Converts each page of a PDF file into separate image files at the specified DPI. 
    The image files are saved in the specified output directory with the original PDF's title plus the page number in the filename. """
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=dpi)
    total_pages = len(images)
    
    progress_bar = tqdm(total=total_pages, unit='page')
    
    for i, image in enumerate(images):
        image_filename = f'{title}_page_{i+1}.jpg'
        image_path = os.path.join(output_directory, image_filename)
        image.save(image_path, 'JPEG')
        progress_bar.update(1) 
    
    progress_bar.close() 

if __name__ == "__main__":
    pdf_path = 'data/Owners_Manual_modelY.pdf'  # Replace with your PDF file path
    output_directory = "data/Owners_Manual_modelY_seperatepagejpgs"  # Replace with your desired output directory
    dpi = 200  # Replace with your desired DPI, for reasonable quality something between 150 and 300 is recommended
    pdf_to_images(pdf_path, output_directory, dpi)