---
layout: post
title: "Suckerbug: A python script to scrape photos from facebook's public pages"  
author: krunal kshirsagar
---
P.S. The recent facebook outage grabbed my attention hence I wrote the script for fun ;p


## Install the required libraries by entering the commands in the command terminal given below:

```
pip install facebook-scraper
pip install Pillow
pip install requests
```
- [facebook-scraper](https://github.com/kevinzg/facebook-scraper) is a great library with which you can do much more than just scrapping photos from facebook.

- [Pillow](https://github.com/python-pillow/Pillow) is a python imaging library.

- [Requests](https://github.com/psf/requests) is an HTTP request library used to process URLs.

## Importing packages
```
from facebook_scraper import *
import os
import io
import requests
from PIL import Image
import tempfile
```
- Now I wanted to make a function that will process the HTTP request, iterate through the page contents and return images that will be stored in the temporary folder(in my case I've saved it in the current directory).

## Download image function

{% highlight ruby %}
def download_image(img_url, out_dir, img_name):
    buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        downloaded = 0
        filesize = int(r.headers['content-length'])
        for chunk in r.iter_content(chunk_size=1024):
            downloaded += len(chunk)
            buffer.write(chunk)
            print(downloaded/filesize)
        buffer.seek(0)
        i = Image.open(io.BytesIO(buffer.read()))
        i.save(os.path.join(out_dir, img_name), quality=85)
    buffer.close()
{% endhighlight ruby %}

## Calling the inbuilt get_photos() function from facebook-scraper library

{% highlight ruby %}
count = 0

for img in get_photos('name of the public page without spaces', pages=1000):
    img_name = "".join(['image', str(count), '.jpg'])
    download_image(img['images'][0], "", img_name)
    count += 1
{% endhighlight ruby %}

Done.

**Please find the script on [Github](https://github.com/Noob-can-Compile/Suckerbug).**

Thanks for reading.