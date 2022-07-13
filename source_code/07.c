#include<stdio.h>
int main()
{
	int W, H, x, y, r;
	r>0;
	scanf("%d %d %d %d %d", &W, &H, &x, &y, &r);
	if(x+r<=W && y+r<=H && 0<x && 0<W && 0<y && 0<H){
		printf("Yes");
	}else if(x+r<=W && y-r>=H && 0<x && 0<W && 0>y && 0>H){
		printf("Yes");
	}else if(x-r<=W && y+r<=H && 0>x && 0>W && 0<y && 0<H){
		printf("Yes");
	}else if(x-r>=W && y-r>=H && 0>x && 0>W && 0>y && 0>H){
		printf("Yes");
	}else{
		printf("No");
	}
	
	return 0;
}