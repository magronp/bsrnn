# BSRNN

<center><a href="https://arxiv.org/pdf/2209.15174.pdf">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png"></a></center>


commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*

Resultats obtenus avec Magbs:
- GRU / LSTM semblent marcher similairement, mais GRU moins de paramètres (7.4M vs. 6.6M pour vocals), donc un peu plus rapide
- conv+conv
- gros modèle (feature_dim=128 / num_reapeat=8)
- gros modèle (feature_dim=64 / num_reapeat=10): pas de diff avec 8
- avec t+mag, semble identitique voire un peu moins bien que sans
