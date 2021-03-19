this code trains an NLP LSTM on a dataset (netflix_titles.csv) of synopsis and descriptions/generes from Netflix shows. The resulting model is
multilabel and can predict the top most likely descriptions/genres for any given any given synopsis. The novelty of this code is that it uses
Tensorflow's TextVectorization layer, which converts each word into a numeric vector. This vectors are an embedded layer of the model and 
change as it trains, such that similar words will result in similar vectors.
