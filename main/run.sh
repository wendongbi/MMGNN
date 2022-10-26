# Dataset_list:
#   * Social Networks: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81,  Flickr, chameleon, squirrel, Tulane29,
#   * Citation Networks: cora, pubmed, citeseer
#   For more information, please refer to our paper: https://arxiv.org/pdf/2208.07012.pdf


# # For social networks
python3 main_citation.py --dataset Simmons81 --hidden 256  --num_epoch 600 --layer 2
# python3 main_citation.py --dataset Hamilton46 --hidden 256  --num_epoch 600 --layer 2
# python3 main_citation.py --dataset GWU54 --hidden 256  --num_epoch 600 --layer 2
# python3 main_citation.py --dataset Howard90 --hidden 256  --num_epoch 600 --layer 2
# python3 main_citation.py --dataset Northeastern19 --hidden 256  --num_epoch 600 --layer 2
# python3 main_citation.py --dataset UGA50 --hidden 256  --num_epoch 600 --layer 2


# # For citation networks
# python3 main_citation.py --dataset pubmed --hidden 256 --lamda 0.4 --wd1 5e-4 --num_epoch 600
# python3 main_citation.py --dataset citeseer --hidden 256  --num_epoch 600
# python3 main_citation.py --dataset cora --test --num_epoch 400 --hidden 64