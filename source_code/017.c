#include<stdio.h>
int main(void)
{
	int W,H,x,y,r,a;
	scanf("%d %d %d %d %d",&W,&H,&x,&y,&r);
	if(x-r<0)
		a=0;
	else if(y-r<0)
		a=0;
	else if(x+r>W)
		a=0;
	else if(y+r>W)
		a=0;
	else
		a=1;
	if(a==1)
		printf("Yes\n");
	else
		printf("No\n");
	return 0;
}