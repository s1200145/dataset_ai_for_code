#include <stdio.h>

int main(){
    int w, h, x, y, r;

    fscanf( stdin, "%d %d %d %d %d", &w, &h, &x, &y, &r);

    if( (x+r)<w && (x-r)>0 && (y+r)<h && (y-r)>0 ){
        printf("Yes\n");
    } else {
        printf("No\n");
    }
    return 0;
}