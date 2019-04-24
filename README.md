# Jigsaw Unintended Bias in Toxicity Comment
---
This repository utilizes the Jigsaw comment data to analyze, visualize, and classify the toxic comments.

### Part 1: EDA on comment text

The data contains the target score (or toxicity score) with other targets, which can be used. However, this repository is focusing on the toxic score / class.

![output 1](https://github.com/netsatsawat/Jigsaw-Toxicity-Comment/blob/master/img/Comment_target_viz.JPG)

The notebook demonstrates the usage of word cloud with masked image. The below code shows the __toxic__ class word cloud.
```python
mask = np.array(Image.open(os.path.join(PATH, "../img/trump2.png")))
wc = WordCloud(background_color="white", max_words=2000, mask=mask,
               contour_width=1, contour_color='grey',
               max_font_size=40, random_state=SEED)

wc.generate_from_frequencies(toxic_freq_dist)
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=(12, 6))
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.title(r"Top words in $\bf{Toxic}$ comments")
plt.show();
```

![output 2](https://github.com/netsatsawat/Jigsaw-Toxicity-Comment/blob/master/img/Toxic_wordcloud.JPG)

Similar coding can be used to visualize the __non toxic__ class word as well.

![output 3](https://github.com/netsatsawat/Jigsaw-Toxicity-Comment/blob/master/img/Non_toxic_wordcloud.JPG)


### Part 2: Model

TBC
