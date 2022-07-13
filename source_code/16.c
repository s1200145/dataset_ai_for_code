#include<stdio.h>

int main(void)
{
	int W, H, x, y, r;
	
	scanf("%d %d %d %d %d", &W, &H, &x, &y, &r);
	
	if((y + r) > H || (x + r) > W || (y - r) < 0 || (x - r) < 0)printf("NO\n");
	else printf("Yes\n");
	
	return 0;
	
}