import imageio
import glob
import numpy as np
import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import sys

jpgfilenames = glob.glob("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/png-nolabelSPP/*.png")
print(jpgfilenames[:5])
jpgfilenames = sorted(jpgfilenames)
print(jpgfilenames[:5])
#! TODO: Sort jpgfilenames before processing 

font_fname = '/home/dwight.velasco/.local/share/fonts/Roboto-MediumItalic.ttf'
# '/home/dwight.velasco/.local/share/fonts/Roboto-MediumItalic.ttf'
# '/usr/share/fonts/open-sans/OpenSans-Italic.ttf'
font_size = 46
font = ImageFont.truetype(font_fname, font_size)

years = [2015,2016,2017,2018]
yearlength = [0,365,731,1096]
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

monthstart = [0,31,59,90,120,151,181,212,243,273,304,334]
leapstart = [0,31,60,91,121,152,182,213,244,274,305,335]

monthlength = [31,59,90,120,151,181,212,243,273,304,334,365]
leaplength = [31,60,91,121,152,182,213,244,274,305,335,366]
"""
for yearidx, year in enumerate(years):
    yearstart = yearlength[yearidx]

    for monthidx, month in enumerate(months):

        if yearidx != 1:
            end = yearstart + monthlength[monthidx]
            start = yearstart + monthstart[monthidx]
        else:
            end = yearstart + leaplength[monthidx]
            start = yearstart + leapstart[monthidx]
        
        print(year, month, end-start, start, end)

        for jpgidx, jpgfilename in enumerate(jpgfilenames[start:end]):
            image = Image.open(jpgfilename)
            draw = ImageDraw.Draw(image)
            x, y = 5, 0
            month = month
            year = year
            date = "%s %d" % (month, year)
            # print(date)
            shadowcolor = "black"
            fillcolor = "#ffffff"

            draw.rectangle(((0, y-2),(245,46)), fill=fillcolor)

            # thicker border
            draw.text((x-2, y-2), date, font=font, fill=shadowcolor)
            draw.text((x+2, y-2), date, font=font, fill=shadowcolor)
            draw.text((x-2, y+2), date, font=font, fill=shadowcolor)
            draw.text((x+2, y+2), date, font=font, fill=shadowcolor)

            draw.text((x, y), date, font=font, fill=fillcolor)

            # //image = image.resize((537,898))
            image.save('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/pngSPP/img-{:04d}.png'.format(start+jpgidx+1))
"""
print("Finished adding text...")

pngfilenames = glob.glob("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/pngSPP/img-*.png")
print(pngfilenames[:2])
pngfilenames = sorted(pngfilenames)
images = []
print(pngfilenames[:2])

# for pngidx, pngfilename in enumerate(pngfilenames):
#     print(pngidx+1)
#     images.append(imageio.imread(pngfilename))
# imageio.mimsave('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/gifs/NCR-movie-blackbg.gif', images, fps=24)


def gen_frame(path):
    im = Image.open(path)
    
    alpha = im.getchannel('A')

    # Convert the image into P mode but only use 255 colors in the palette out of 256
    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
    # Set all pixel values below 128 to 255 , and the rest to 0
    mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)

    # Paste the color of index 255 and use alpha as a mask
    im.paste(255, mask)

    # The transparency index is 255
    im.info['transparency'] = 255

    return im

for pngidx, pngfilename in enumerate(pngfilenames):
    print(pngidx+1)
    frame = gen_frame(pngfilename)
    images.append(frame.copy())
    frame.close()

images[0].save('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/splitbands/gifs/NCR-movie-transparent.gif', save_all=True, format='GIF', append_images=images[1:], fps=24, disposal=0, loop=0, transparency=255)
