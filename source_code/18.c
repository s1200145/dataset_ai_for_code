#include<stdio.h>
int main(void){
  int w,h,x,y,r;
  scanf("%d %d %d %d %d",&w,&h,&x,&y,&r);
  if((y-r)<0)printf("NO\n");
  else if((x-r)<0)printf("NO\n");
  else if(h<(y+r))printf("NO\n");
  else if(w<(x+r))printf("NO\n");
  else printf("YES\n");
  return 0;
}