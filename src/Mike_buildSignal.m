function x = buildSignal(theta,N,type);

if strcmp(type,'cosine')
  x = cos([1:N]'*theta*pi);
elseif strcmp(type,'gaussian')
  x = exp(-([1:N]'-N*theta).^2/2/N);
elseif strcmp(type,'step')
  x = zeros(N,1);
  transition = ceil(theta*N);
  transVal = ceil(theta*N)-theta*N;
  if transition < N
    x(transition+1:N) = 1;
  end
  if (transition > 0)
    x(transition) = transVal;
  end
end 