clear
clc
Table = readtable('..\Atlas\Fan246\BNA_subregions.xlsx');
L = Table.LabelID_L;
R = Table.LabelID_R;

idx = [1, 35, 63, 82, 88, 95, 106];
label = Table.Lobe(idx);
a = flipud(L(idx));
b = R(idx);
A = [a;b(2:end)];

BN_idx = [flipud(L); R];

boundary_idx = zeros(length(A),1);
for i=1:length(A)
    boundary_idx(i) = find(BN_idx==A(i));
end
boundary_idx(8:end)=boundary_idx(8:end)-1;

save('my_BN_order.mat','BN_idx', 'boundary_idx');