---
layout: post
title: "How to build a Reddit bot to scrape saved posts by filtering out NSFW posts"  
author: krunal kshirsagar
---
This script filters out the NSFW posts and scrapes only the Non-NSFW posts which are stored in `saved posts` of your Reddit account into a `.csv` file.  
As I was implementing this script, I realized that my code isn’t universal. Therefore, I would like to provide you with my code and in order to get the script running for you, you need to tweak some class names or element id according to your Reddit account via CSS query selector by inspecting elements in your browser.  


However, Do run the script initially by inserting your credentials wherever asked. for example wherever there is `Your-username` or `Your-password` written in the script, do insert your Reddit **_username/email id_** and **_password_** there.  


### Here’s how to do it:
1. You should have selenium web driver installed in your python environment, To install selenium enter below command in the command prompt or anaconda prompt or vs code terminal:  
**`pip install selenium`**  
2. After installing selenium, make sure you have a firefox browser installed on your system.  
3. You need some knowledge of CSS query selector in order to access the elements via class name on Reddit by inspecting element on the browser(firefox).   
4. The code isn’t universal so you need to tweak the code according to your Reddit Classnames by inspecting elements and using CSS query selectors on the browser.  

Checkout the code on **[Github](https://github.com/Noob-can-Compile/reddit_saved_post_filter)**

