#include <stdio.h>

int main(void)
{
  int w, h, x, y, r;
  scanf("%d %d %d %d %d", &w, &h, &x, &y, &r);
  puts(( (((x-r)<0) && ((x+r)>w)) || (((y-r)<0) && ((y+r)>h)) ) ? "No" : "Yes");
  return 0;
}