from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset="all",
    categories=["sci.space", "rec.sport.hockey", "talk.politics.guns", "rec.autos"],
)

print(newsgroups.data[0])
