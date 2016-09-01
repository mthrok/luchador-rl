set -e

sudo apt-get update
sudo apt-get install -y git libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake

git clone https://github.com/mgbellemare/Arcade-Learning-Environment
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make
pip install --upgrade .