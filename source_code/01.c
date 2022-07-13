#include <stdio.h>

int main(void)
{
  int W, H, x, y, r;

  scanf("%d %d %d %d %d",&W,&H,&x,&y,&r);

  if(((0 <= y - r) && (r < H  - y)) && ((0 <= x - r) && (r < W - x)))
	printf("Yes\n");
  else
	printf("No\n");

  return 0;
}