#include <stdio.h>

int main(int argc, char **argv)
{
    int w,h,x,y,r;

    scanf("%d %d %d %d %d", &w, &h, &x, &y, &r);
    if ((x >= r) && (x <= w - r) &&             
        (y >= r) && (y <= h - r)) {             
        printf("Yes\n");
    } else printf("Nes\n");

    return 0;
}