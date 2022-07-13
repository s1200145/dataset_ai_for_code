#include <stdio.h>

int main()
{
    int rect_w, rect_h, x, y, r;

    scanf("%d %d %d %d %d", &rect_w, &rect_h, &x, &y, &r);

    if (x > r && x + r < rect_w && y > r && y + r < rect_h){
        printf("Yes\n");
    }else{
        printf("No\n");
    }

    return 0;
}