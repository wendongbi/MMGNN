# Dataset_list:
#   * Social Networks: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81,  Flickr, chameleon, squirrel, Tulane29,
#   * Citation Networks: cora, pubmed, citeseer
#   * Webpage Networks: chameleon, squirrel
#   * Other Networks: Flickr, etc.
#   For more information, please refer to our paper: https://arxiv.org/pdf/2208.07012.pdf



# # For citation networks
python3 main.py --dataset cora --test --num_epoch 400 --hidden 64
# python3 main.py --dataset pubmed --hidden 256 --lamda 0.4 --wd1 5e-4 --num_epoch 600
# python3 main.py --dataset citeseer --hidden 256  --num_epoch 600