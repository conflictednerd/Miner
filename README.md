# Miner
A command line program to suggest interesting papers from arxiv

To start, run the `app.py` file: `python app.py`.

`data.txt, dataset.pkl, nn.pth, optim.pth, authors.json` are populated based on my own taste. To train the recommender model with your own data, you need an initial dataset of arxiv links and labels (0 for uninteresting, 1 for interesting). Look at the `data.txt` file to see how this dataset should be. Furthermore, you can specify some authors that you like (or hate) in the `authors.json` file so that they can be identified and used when suggesting papers.
Once you have this dataset, you should remove the old `dataset.pkl`, `nn.pth`, `optim.pth` files and keep only the updated `data.txt` and `authors.json` files. To train the model with this initial dataset, run the app and select Expand Dataset from the main menu. Once that's done, select train Train Model from the main menu and train the model for some epochs (10 is good). After training is done, you can choose to save your model.

To get suggestions, select the appropriate option from the main menu. As you rate more papers, the dataset will expand, and you can further train your model with the newly collected data after some time.

Notice: Right now when you select "Suggest Papers" from the main menu, it takes a while to fetch and process new papers from arxiv. This is normal, and is due to the fact that there isn't any multi-processing or proper batching implemented. just give it some time.

Notice: The first time you run the program, a ~400MB language model from Hugging face will be downloaded, which is used to process the papers. This happens only the first time you run the program, and from then on a cached version will be used.
