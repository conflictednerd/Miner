import json
import pickle
import random
import time

import arxiv
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import just_fix_windows_console
from prettytable import PrettyTable
from rec_models import NN
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from termcolor import cprint
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


class MinerApp:
    def __init__(self):
        self._dataset_file = "dataset.pkl"
        self._info_dataset_file = "info.pkl"
        self._model_file = "nn.pth"
        self._optim_file = "optim.pth"
        self._author_file = "authors.json"
        self.lr = 1e-3
        self.batch_size = 8
        just_fix_windows_console()

    def start(self):
        print("Loading models...")
        self.emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.nn_model = NN()
        try:
            self.nn_model.load_state_dict(torch.load(self._model_file))
        except:
            pass
        with open(self._author_file) as f:
            known_authors = json.load(f)
            self.known_authors = [a.lower() for a in known_authors]
        print(
            "Welcome to Miner, a not-yet-tested command line application that suggests papers based on your taste!"
        )
        self.main_menu()

    def main_menu(self):
        print("Main Menu")
        print("Choose what you want to do next:\n")
        menu = PrettyTable()
        menu.field_names = ["Option", "Description"]
        menu.add_row(["1", "Get Suggestions"])
        menu.add_row(["2", "Train Model"])
        menu.add_row(["3", "Expand Dataset"])
        menu.add_row(["4", "Exit"])
        print(menu)
        choice = input("Enter your choice (1-4): ")
        if choice == "1":
            self.suggest_menu()
        elif choice == "2":
            self.train_menu()
        elif choice == "3":
            self.expand_dataset_menu()
        elif choice == "4":
            print("Goodbye!")
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            self.main_menu()

    def suggest_menu(self):
        papers = self._fetch(n=500)
        # create an ascending list according to score and another according to distance from 0.5 score
        # randomly select either a good (0.5 chance), bad (0.25), or average (0.25) paper to suggest.
        good_papers = sorted(
            papers, key=lambda p: p["score"], reverse=True
        )  # from good to bad
        mid_papers = sorted(papers, key=lambda p: abs(p["score"] - 0.5))
        g, b, m = 0, -1, 0  # index of next good, bad, or mid
        dataset = []
        info_dataset = []
        _continue = True
        while _continue:
            r = random.uniform(0, 1)
            if r < 0.7:
                paper = good_papers[g]
                g += 1
            elif r < 0.8:
                paper = good_papers[b]
                b -= 1
            else:
                paper = mid_papers[m]
                m += 1
            self._display_paper(paper)
            valid_choice = False
            while not valid_choice:
                choice = input(
                    "Was it good? Type y for Yes, n for No, a to show the abstract, s to skip, and e for Exit.\n"
                )
                if choice.lower() in ["y", "yes"]:
                    dataset.append((paper["emb"], 1))
                    paper["label"] = 1
                    info_dataset.append(paper)
                    valid_choice = True
                elif choice.lower() in ["n", "no"]:
                    dataset.append((paper["emb"], 0))
                    paper["label"] = 0
                    info_dataset.append(paper)
                    valid_choice = True
                elif choice.lower() in ["a", "abs", "abstract"]:
                    cprint(f"Abstract: {paper['summary']}", color="yellow")
                    print("=" * 60)
                elif choice.lower() in ["s", "skip"]:
                    valid_choice = True
                elif choice.lower() in ["e", "exit"]:
                    if len(dataset) > 0:
                        self._add_to_dataset(dataset)
                        self._update_info_dataset(info_dataset)
                    valid_choice = True
                    _continue = False
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
        self.main_menu()

    def train_menu(self):
        try:
            ds = self._load_dataset()
        except:
            print(
                'No dataset was found. Either choose "Expand Dataset" from the main menu and use a data.txt file, or rate some suggestions.'
            )
            self.main_menu()
        epochs = int(input("Epochs: "))
        self._train(ds, epochs)
        self.main_menu()
        # todo store n previous versions + the latest

    def expand_dataset_menu(self):
        dataset = []
        info_dataset = []
        file_name = input(
            'Input the name of a txt file that contains -on each line- an arxiv link and a 0-1 label separated by a single space (e.g., "data.txt"): '
        )
        with open(file_name, "r") as f:
            lines = f.readlines()
        for i, line in tqdm(enumerate(list(lines)), desc="Papers"):
            url, label = line.split(" ")
            for try_count in range(5):
                try:
                    paper = self._get_paper_from_url(url)
                    paper["label"] = int(label)
                    dataset.append((paper["emb"], int(label)))
                    info_dataset.append(paper)
                    break
                except Exception as e:
                    time.sleep(2)
                    if try_count == 4:
                        print(f"Fetching index {i} failed ({url})")
                        print(e)
        self._add_to_dataset(dataset)
        self._update_info_dataset(info_dataset)
        print("Done!")
        self.main_menu()

    @torch.no_grad()
    def _fetch(self, n):
        print("Fetching from arxiv...")
        client = arxiv.Client()
        search = arxiv.Search(
            query="cat:cs.SY OR cat:cs.RO OR cat:cs.OH OR cat:cs.NE OR cat:cs.MA OR cat:cs.LG OR cat:cs.IT OR cat:cs.HC OR cat:cs.GT OR cat:cs.CV OR cat:cs.CL OR cat:cs.AI",
            max_results=n,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers = [
            {
                "title": paper.title,
                "link": paper.entry_id,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
            }
            for paper in arxiv.Client().results(search)
        ]
        print("Processing data...")
        text_embeddings = self.emb_model.encode(
            [
                f'Title: "{paper["title"]}"\nSummary:\n{paper["summary"]}'
                for paper in papers
            ],
            convert_to_numpy=False,
        )

        papers_with_embeddings = []
        for paper, text_emb in zip(papers, text_embeddings):
            author_emb = torch.zeros(256, dtype=text_emb.dtype)
            author_indices = [
                self.known_authors.index(author.lower())
                if author.lower() in self.known_authors
                else -1
                for author in paper["authors"]
            ]
            for i in author_indices:
                if i >= 0:
                    author_emb[i] = 1
            paper_embedding = torch.cat([text_emb.flatten().detach(), author_emb])
            score = torch.sigmoid(self.nn_model(paper_embedding)).item()
            paper["emb"] = paper_embedding
            paper["score"] = score
            papers_with_embeddings.append(paper)
        print("Completed!")
        return papers_with_embeddings

    @torch.no_grad()
    def _get_paper_from_url(self, url):
        # get the paper from arxiv and create its dictionary and embedding
        client = arxiv.Client()
        search = arxiv.Search(
            id_list=[
                url.split("/")[-1],
            ]
        )
        paper = list(arxiv.Client().results(search))[-1]
        paper = {
            "title": paper.title,
            "link": paper.entry_id,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
        }
        text_emb = self.emb_model.encode(
            [
                f'Title: "{paper["title"]}"\nSummary:\n{paper["summary"]}',
            ],
            convert_to_numpy=False,
        )[0].detach()
        author_emb = torch.zeros(256, dtype=text_emb.dtype)
        author_indices = [
            self.known_authors.index(author.lower())
            if author.lower() in self.known_authors
            else -1
            for author in paper["authors"]
        ]
        for i in author_indices:
            if i >= 0:
                author_emb[i] = 1
        paper_embedding = torch.cat([text_emb.flatten(), author_emb])
        paper["emb"] = paper_embedding
        return paper

    def _display_paper(self, paper):
        print("=" * 30)
        cprint(f"Title: {paper['title']}", color="light_blue", attrs=["bold"])
        cprint(f"Authors: {', '.join(paper['authors'])}", color="light_cyan")
        cprint(f"Score: {paper['score']:.2f}", color="light_magenta")
        cprint(f"Link: {paper['link']}", color="red")
        print("=" * 30)

    def _add_to_dataset(self, papers):
        # papers is a list of <embedding, label> tuples
        try:
            with open(self._dataset_file, "rb") as f:
                db = pickle.load(f)
        except:
            db = []
        finally:
            db.extend(papers)
            with open(self._dataset_file, "wb") as f:
                pickle.dump(db, f)
            print(f"Current Dataset size is {len(db)}.")

    def _update_info_dataset(self, papers):
        # papers is a list of dicts
        try:
            with open(self._info_dataset_file, "rb") as f:
                db = pickle.load(f)
        except:
            db = []
        finally:
            db.extend(papers)
            with open(self._info_dataset_file, "wb") as f:
                pickle.dump(db, f)

    def _load_dataset(self):
        with open(self._dataset_file, "rb") as f:
            dataset = pickle.load(f)
        return TensorDataset(
            torch.stack([x[0] for x in dataset]),
            torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in dataset]),
        )

    def _train(self, dataset, epochs, train_on_all=False, log_interval=6):
        if not train_on_all:
            t_size = int(len(dataset) * 0.75)
            v_size = len(dataset) - t_size
            train_set, val_set = random_split(dataset, [t_size, v_size])
            train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, self.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(dataset, self.batch_size, shuffle=True)
            val_loader = None

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.lr)
        try:
            optimizer.load_state_dict(torch.load(self._optim_file))
        except:
            pass

        for epoch in range(epochs):
            self.nn_model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.nn_model(data)
                loss = criterion(output.squeeze(), target.float())
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.3f}"
                    )

            self._validate_model(
                val_loader if val_loader is not None else train_loader,
                f"Epoch {epoch+1}",
            )

        self._validate_model(
            DataLoader(dataset, self.batch_size, shuffle=False), "Whole Dataset"
        )

        save = input("Save? (y for Yes, n for No)\n").lower() in ["y", "yes"]
        if save:
            torch.save(self.nn_model.state_dict(), self._model_file)
            torch.save(optimizer.state_dict(), self._optim_file)

    def _validate_model(self, loader, epoch_name):
        self.nn_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                output = self.nn_model(data)
                predictions = (output >= 0.0).float()

                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)

        print(f"\n{epoch_name} - Validation Metrics:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")


if __name__ == "__main__":
    app = MinerApp()
    app.start()
