function [x] = normalize(x)
    a = prctile(x(:),98);
    x = x/a;
    x(find(x<-1))=-1;
    x(find(x>1))=1;
end

