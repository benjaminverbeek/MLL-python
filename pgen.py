
def prime_generator():
    primes = [2]
    yield 2
    i = 2
    while True:
        i += 1
        for p in primes:
            if i % p == 0:
                break
            elif p > i**0.5:
                primes.append(i)
                yield i
                break

gen = prime_generator()

for i in range(100):
    print(next(gen))