clear all;
img_in=imread("E:\Pycharm\projects\Test_n\results\result_0.png");
%灰度化
img_gray=im2gray(img_in);%subplot(3,3,2);
img1=img_gray;
 %二值化
img1=im2bw(img1,graythresh(img1));
%figure;imshow(img1);
%img_before=~img_in; 
%图片处理后即为黑底，无需反转
%figure;imshow(img_before);
%为了保证做极限腐蚀时不受边界突变凹坑的影响，先对整幅图像做一个边缘的光滑处理。 
%这里使用膨胀。这样做的话极限腐蚀后最终腐蚀的区域数只有三个，消去了由于边界不光 
%滑造成的影响。 
img=imdilate(img1,ones(8,8));%使用膨胀法进行边界平滑。 
% 
%figure;imshow(img);
%img=img_before;
[m,n]=size(img); 
imgn=zeros(m,n); 
preimg=imgn; 
% 
se=strel('square',3);%stricture element. 
 
%开始 
while sum(sum(preimg-img))~=0  
    preimg=img; 
    img=img>0; 
    [img label]=bwlabel(img);      %标记不同区域，label是区域个数 
     
    imgn=imerode(img,se); 
    %腐蚀之后是否有哪个被标记的区域消失了 
    Hist=zeros(1,label);             
    for i=1:m 
        for j=1:n 
            if imgn(i,j)~=0 
                Hist(imgn(i,j))=imgn(i,j);   
            end 
        end 
    end 
    %统计消失区域的标号 
    H=[]; 
    for i=1:label 
        if Hist(i)==0 
            H=[H i];        
        end 
    end 
    %如果这个区域消失了，那么再把这个区域恢复过来 
    if ~isempty(H) 
        l=length(H); 
        for i=1:m 
            for j=1:n 
                for k=1:l 
                    if img(i,j)==H(k)    
                        imgn(i,j)=img(i,j); 
                    end 
                end 
            end 
        end    
    end     
    img=imgn; 
end 
a=label  %输出区域数量 
 
figure
imshow(img);title('最终腐蚀结果——“核”');
