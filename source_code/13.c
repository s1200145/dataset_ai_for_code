#include <stdio.h>

int main(void) {
	int W, H, x, y, r;
	
	(void)scanf("%d %d %d %d %d", &W, &H, &x, &y, &r);
	
	/* x -direction */
	if ( (0 < x-r) && (x+r < W) && (0 < y-r) && (y+r < H) ) {
		printf("Yes\n");
	} else {
		printf("No\n");
	}
	
	return 0;
}