#include <stdio.h>
int main()
{
    int W;
    int H;
    int x;
    int y;
    int r;
          
    scanf("%d %d %d %d %d",&W,&H,&x,&y,&r);
         
    if(0<=x-r&&W>=x+r&&0<=y-r&&W>=y+r){
      puts("Yes");
    }
    else{
      puts("No");
    }
    return 0;
}