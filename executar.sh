rm -rf *.rst

for imagem in 04
do
	for threads in 2 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69
	do
		mpirun -np $threads --hostfile hosts-$threads ./convolution images/$imagem.jpg out.jpg 0
	done
done
