
commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*


Note : differences since the model def need to be slightly adapted to output also the STFT (needed for the original training loss)


Vieilles remarques sur MAGBS, à retenter ici sur le BSRNN (lstm vs gru vs conv ; attention ; domain pour la loss)

v0/v1 : LSTM (OLD)
v2/v3 : GRU (OLD)
v8/v11: GRU time + CONV band , mais l'output correspond au training pas fini (v8)

Après ajout de RELU en sortie de masquage pour forcer spectro nonnegatif, et avec GRU+GRU, feature_dim=64, num_repeat=8

v18/24: n_att_head=0..
v19:    n_att_head=4
v22/23: n_att_head=0, loss_domain=t+mag

Ne marche pas bien :
- conv2D
- conv+conv
- gros modèle (feature_dim=128 / num_reapeat=8)
- gros modèle (feature_dim=64 / num_reapeat=10): pas de diff avec 8
- avec t+mag, semble identitique voire un peu moins bien que sans

GRU / LSTM semblent marcher similairement, mais GRU moins de paramètres (7.4M vs. 6.6M pour vocals), donc un peu plus rapide
