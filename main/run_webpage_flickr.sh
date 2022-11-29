# Dataset_list:
#   * Social Networks: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81,  Flickr, chameleon, squirrel, Tulane29,
#   * Citation Networks: cora, pubmed, citeseer
#   * Webpage Networks: chameleon, squirrel
#   * Other Networks: Flickr, etc.
#   For more information, please refer to our paper: https://arxiv.org/pdf/2208.07012.pdf


# # For webpage networks and Flickr network
python3 main.py --dataset squirrel --hidden 256  --num_epoch 400 --layer 2
# python3 main.py --dataset chameleon --hidden 256  --num_epoch 400 --layer 2 
# python3 main.py --dataset Flickr --hidden 256  --num_epoch 400 --layer 2
