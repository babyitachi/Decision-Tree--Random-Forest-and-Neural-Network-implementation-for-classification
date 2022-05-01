#!/bin/bash 
if [ -z $1 ];
	then echo "Please provide appropriate argument";
	exit;
fi;
if [ $1 -eq 1 ];
	then if [ -z $5 ];
		 then echo "Invaild parameters for Q1";
		 else python3 Q1/Q1.py $2 $3 $4 $5;
		 fi;
	elif [ $1 -eq 2 ]	
	then if [ -z $4 ];
		 then echo "Invaild parameters for Q2";
		 else python3 Q2/Q2.py $2 $3 $4;
		 fi;
else echo "Invalid Question number. Please provide valid Question number (1 or 2).";
fi;