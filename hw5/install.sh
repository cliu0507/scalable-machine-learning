module load gcc;
module swap intel gcc/4.7.1;
module load cmake;
export BOOST\_ROOT=/opt/apps/gcc4\_7/boost/1.55.0/;
tar xvzf Galois-2.2.1.tar.gz;
cd Galois-2.2.1/build;
mkdir default; cd default;
cmake ../..; make;
