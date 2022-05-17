#include <iostream>

int main()
{
    std::string str, str2;
    int a = 25;
    getline(std::cin, str);


    std::cout << str.length() << std::endl;

    str2 = str;

    std::cout << str2.length() << std::endl;

    str += " naum";
    std::cout << str << std::endl;

    return 0;
}