
commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*

jobs à lancer :

Impact de la loss (sur modèle moyen)

./book_train
./book_train 'optim.loss_domain=[t] src_mod.time_layer=conv'
./book_train 'optim.loss_domain=[tf] src_mod.time_layer=conv'

LSTM vs. GRU vs. CONV (loss temporelle, modèle moyen), d'abord changer time_layer, puis band_layer

./book_train 'optim.loss_domain=[t] src_mod.time_layer=gru'
./book_train 'optim.loss_domain=[t] src_mod.time_layer=conv'
./book_train 'optim.loss_domain=[t] src_mod.band_layer=gru'
./book_train 'optim.loss_domain=[t] src_mod.band_layer=conv'

Taille du modèle (loss temporelle, lstm+lstm)

./book_train 'optim.loss_domain=[t] src_mod.num_reapeat=10' gruss
./book_train 'optim.loss_domain=[t] src_mod.feature_dim=128' gruss

Attention (loss temporelle, lstm+lstm)

./book_train 'optim.loss_domain=[t] src_mod.n_att_head=1'
./book_train 'optim.loss_domain=[t] src_mod.n_att_head=2 src_mod.attn_enc_dim=10'
./book_train 'optim.loss_domain=[t] src_mod.n_att_head=2 src_mod.attn_enc_dim=20'



Resultats obtenus avec Magbs:
- GRU / LSTM semblent marcher similairement, mais GRU moins de paramètres (7.4M vs. 6.6M pour vocals), donc un peu plus rapide
- conv+conv
- gros modèle (feature_dim=128 / num_reapeat=8)
- gros modèle (feature_dim=64 / num_reapeat=10): pas de diff avec 8
- avec t+mag, semble identitique voire un peu moins bien que sans
