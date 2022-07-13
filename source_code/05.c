#include<stdio.h>
int main(){
	int w,h,x,y,r;
	scanf("%d%d%d%d%d",&w,&h,&x,&y,&r);
		if(x<=0||x>=w||x+r>w||x-r<0||y<=0||y>=h||y+r>h||y-r<0){
			printf("No");
			}else{printf("Yes");}
	return 0;
}