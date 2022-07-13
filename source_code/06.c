#include<stdio.h>
#include <stdlib.h>

int main(){

int W, H, x, y, r;

scanf("%d", &W);
scanf("%d", &H);
scanf("%d", &x);
scanf("%d", &y);
scanf("%d", &r);

if( x+r<W && x-r>0 && y+r<H && y-r>0 ){
   printf("Yes\n");
} else {
   printf("No\n");
}


return 0;
}