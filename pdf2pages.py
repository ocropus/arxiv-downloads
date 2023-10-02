import webdataset as wds
import glob
import os
import re
import msgpack as mp
import tempfile
from itertools import islice


import pdfminer3
from pdfminer3.layout import LAParams, LTTextBox, LTChar
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.pdfdocument import PDFDocument
from pdfminer3.pdfparser import PDFParser
import numpy as np

def bbox_union(bbox1, bbox2):
    x0 = min(bbox1[0], bbox2[0])
    y0 = min(bbox1[1], bbox2[1])
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[3], bbox2[3])

    return (x0, y0, x1, y1)

def bbox_scale(bbox, scale, mediabox):
    llx, lly, urx, ury = mediabox
    x0, y0, x1, y1 = tuple([b*scale for b in bbox])
    result = [x0, ury*scale-y1, x1, ury*scale-y0]
    result = [int(x) for x in result]
    return tuple(result)

def extract_word_bounding_boxes(file, scale=300.0/72.0):
    fp = open(file, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)

    if not document.is_extractable:
        raise pdfminer3.PDFTextExtractionNotAllowed

    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        layout = device.get_result()

        for lt_obj in layout:
            if isinstance(lt_obj, LTTextBox):
                for text_line in lt_obj:
                    words = []
                    # Initialize word and its bounding box
                    word = []
                    bbox = None

                    # Get all characters in a list
                    characters = [character for character in text_line if isinstance(character, LTChar)]
                    spacings = []
                    heights = []

                    # Compute all spacings
                    for i in range(len(characters) - 1):
                        spacings.append(characters[i + 1].bbox[0] - characters[i].bbox[2])
                    for c in characters:
                        heights.append(c.bbox[3] - c.bbox[1])

                    if len(spacings) == 0:
                        continue

                    # Calculate the median of the spacings
                    median_space = np.median(spacings)
                    median_height = np.median(heights)

                    threshold = max(2 * median_space, 0.15 * median_height)

                    for i, character in enumerate(characters):
                        # Append the character to the word
                        word.append(character.get_text())

                        # Update the bounding box of the word
                        if bbox is None:
                            bbox = character.bbox
                        else:
                            bbox = bbox_union(bbox, character.bbox)

                        # If the space after the character is larger than twice the median space,
                        # output the word bounding box
                        if i < len(characters) - 1 and spacings[i] > threshold:
                            words.append(dict(text="".join(word), bbox=bbox_scale(bbox, scale, page.mediabox)))
                            word = []
                            bbox = None

                    # If there is still a word left at the end of the line, output it
                    if word:
                        words.append(dict(text="".join(word), bbox=bbox_scale(bbox, scale, page.mediabox)))
                yield words


