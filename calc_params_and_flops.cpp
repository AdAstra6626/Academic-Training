// ConsoleApplication6.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <typeinfo>
#include <windows.h>
#include <string>
#include <vector>
using namespace std;

void append_(vector<vector<int>> &a, vector<vector<int>> &b) {
    for (int i = 0; i < b.size(); i++) {
        a.push_back(b[i]);
    }
}
int conv_param(int k, int cout, int cin) {
    return k * k * cout * cin + cout;
}

int conv_operate(int k, int cout, int cin, int w, int h) {
    return k * k * cin * cout * w * h;
}

int calc_param_conv(vector<vector<int>> v) {
    int size = v.size();
    int total_param = 0;
    for (int i = 0; i < size; i++) {
        total_param += conv_param(v[i][0], v[i][1], v[i][2]);
    }
    return total_param;
}

int calc_operate_conv(vector<vector<int>> v) {
    int size = v.size();
    int total_operate = 0;
    for (int i = 0; i < size; i++) {
        total_operate += conv_operate(v[i][0], v[i][1], v[i][2], v[i][3], v[i][4]);
    }
    return total_operate;
}

vector<vector<int>> gen_resblock(int cout, int cin, int w, int h) {
    vector<vector<int>> resblock;
    resblock.push_back({ 3,cout,cin,w,h });
    resblock.push_back({ 3,cout,cout,w,h });
    return resblock;
    
}
vector<vector<int>> gen_resnet(vector<int> net) {
    vector<vector<int>> net_;
    net_.push_back({ 3,16,3,32,32 });
    for(int i =0;i<net[0];i++){
        vector<vector<int>> resblock = gen_resblock(16, 16, 32, 32);
        for (int j = 0; j < resblock.size(); j++) {
            net_.push_back(resblock[j]);
        }
    }
    vector<vector<int>> resblock1 = gen_resblock(16, 32, 16, 16);
    for (int j = 0; j < resblock1.size(); j++) {
        net_.push_back(resblock1[j]);
    }
    for (int i = 0; i < net[1]-1; i++) {
        vector<vector<int>> resblock = gen_resblock(32, 32, 16, 16);
        for (int j = 0; j < resblock.size(); j++) {
            net_.push_back(resblock[j]);
        }
    }
    vector<vector<int>> resblock2 = gen_resblock(32, 64, 8, 8);
    for (int j = 0; j < resblock2.size(); j++) {
        net_.push_back(resblock2[j]);
    }
    for (int i = 0; i < net[2] - 1; i++) {
        vector<vector<int>> resblock = gen_resblock(64, 64, 8, 8);
        for (int j = 0; j < resblock.size(); j++) {
            net_.push_back(resblock[j]);
        }
    }
    return net_;
}
vector<vector<int>> vgg16 = { {3,64,3,32,32},
    {3,64,64,32,32},{3,128,64,16,16},{3,128,128,16,16},{3,256,128,8,8},{3,256,256,8,8},{3,256,256,8,8},
    {3,512,256,4,4},{3,512,512,4,4},{3,512,512,4,4},{3,512,512,2,2},{3,512,512,2,2},{3,512,512,2,2} };

vector<vector<int>> gen_bottleneck(int cout, int cin, int w, int h) {
    int expansion = 4;
    vector<vector<int>> result;
    result.push_back({1, 4*cout,cin,w,h});
    result.push_back({ 3, cout,4*cout,w,h });
    return result;

}

vector<vector<int>> gen_denseblock(int initin, int num_bottleneck, int growthrate, int w, int h) {
    vector<vector<int>> result;
    for (int i = 1; i < num_bottleneck; i++) {
        vector<vector<int>> temp = gen_bottleneck(growthrate, initin + i * growthrate, w, h);
        result.insert(result.end(), temp.begin(), temp.end());

    }
    return result;
}

vector<vector<int>> gen_transition(int initin, int w, int h) {
    return { {1, initin, initin / 2, w, h} };
}

vector<vector<int>> gen_densenet(int num_blocks, int &n) {
    vector<vector<int>> result, temp;
    result.push_back({ 3, 24, 3, 32, 32 });
    int initin = 24;
    temp = gen_denseblock(24, num_blocks, 12, 32, 32);
    result.insert(result.end(), temp.begin(), temp.end());
    initin = initin + num_blocks * 12;
    temp = gen_transition(initin, 32, 32);
    initin = initin / 2;

    temp = gen_denseblock(initin, num_blocks, 12, 16, 16);
    result.insert(result.end(), temp.begin(), temp.end());
    initin = initin + num_blocks * 12;
    temp = gen_transition(initin, 16, 16);
    initin = initin / 2;
    temp = gen_denseblock(initin, num_blocks, 12, 8, 8);
    result.insert(result.end(), temp.begin(), temp.end());
    n = initin + num_blocks * 12;
    return result;
}


int main()
{
    int vgg16_pm, vgg16_op;
    vgg16_pm = calc_param_conv(vgg16)+5120;
    vgg16_op = calc_operate_conv(vgg16)+5120;
    cout << "vgg16pm "<<vgg16_pm << endl;
    cout << "vgg16pm " << vgg16_op << endl;
    vector<int> resnet32 = { 5,5,5 };
    vector<int> resnet44 = { 7,7,7 };
    vector<int> resnet110 = { 18,18,18 };
    int resnet32_pm, resnet32_op, resnet44_pm, resnet44_op, resnet110_pm, resnet110_op;
    resnet32_pm = calc_param_conv(gen_resnet(resnet32))+640;
    resnet44_pm = calc_param_conv(gen_resnet(resnet44))+640;
    resnet110_pm = calc_param_conv(gen_resnet(resnet110)) + 640;
    resnet32_op = calc_operate_conv(gen_resnet(resnet32)) + 640;
    resnet44_op = calc_operate_conv(gen_resnet(resnet44)) + 640;
    resnet110_op = calc_operate_conv(gen_resnet(resnet110)) + 640;
    cout << "resnet32 " << resnet32_pm <<" "<< resnet32_op << endl;
    cout << "resnet44 " << resnet44_pm <<" "<< resnet44_op << endl;
    cout << "resnet110 " << resnet110_pm <<" "<< resnet110_op << endl;

    int n, densenet_pm, densenet_op;
    vector<vector<int>> densenet = gen_densenet(12, n);
    densenet_pm = calc_param_conv(densenet);
    densenet_op = calc_operate_conv(densenet);
    cout << "densenet40 " << densenet_pm+n*10 <<" "<< densenet_op+n*10 << endl;


}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
