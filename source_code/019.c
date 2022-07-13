#include <stdio.h>

int main(void){
	
	int W, H, x, y, r;
	
	do {
		scanf("%d %d %d %d %d",&W, &H, &x, &y, &r);
	} while( (x<-100||x>100) || (y<-100||y>100) || (W<=0||W>100) || (H<=0||H>100) || (r<=0||r>100) );
	
	if ( (x>0 && x<W) && (y>0 && y<H) && ((x-r)>0 && (x+r)<W) && ((y-r)>0 && (y+r)<H)) printf("Yes\n");
	else printf("No\n");
	
	return 0;
}