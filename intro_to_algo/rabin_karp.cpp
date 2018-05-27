/**
 * @file   rabin_karp.cpp
 * @author knuu
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>

/**
 * rabin-karp algorithm
 * string-matching algorithm using hashing, chap 32.2 (p.823~)
 * time complexity: let n = |text| and m = |pattern|, O((n + m)n)
 *
 * @param text target text for matching
 * @param dict dictionary for char to int
 * @param pattern matching pattern
 * @param base the number of the kind of input character
 * @param mod prime for hashing
 *
 * @return list of indices of valid shifts
 */
std::vector<int> rabin_karp(std::string text, std::map<char, int> dict, std::string pattern, long long base, long long mod) {
  int n = text.size(), m = pattern.size();
  long long p = 0, t = 0, powd = 1;
  for (int i = 0; i < m; i++) {
    p = (p * base + dict[pattern[i]]) % mod;
    t = (t * base + dict[text[i]]) % mod;
    powd = powd * base % mod;
  }

  std::vector<int> ret;
  for (int i = 0; i < n - m + 1; i++) {
    if (p == t and pattern == text.substr(i, m)) {
      ret.emplace_back(i);
    }
    if (t < n - m) {
      t = (t - powd * dict[i] % mod + mod + dict[i + m]) % mod;
    }
  }
  return ret;
}

int main() {
  const long long base = 26, mod = 1e9 + 7;
  std::map<char, int> dict;
  for (int i = 0; i < 26; i++) {
    dict['a' + i] = i;
  }
  std::string a = "abracadabra";
  std::string b = "abra";

  std::cout << a << ' ' << b << std::endl;
  for (int i : rabin_karp(a, dict, b, base, mod)) {
    std::cout << i << std::endl;
  }

  a = "aaaaaaaaaaaaa";
  b = "aaa";
  std::cout << a << ' ' << b << std::endl;
  for (int i : rabin_karp(a, dict, b, base, mod)) {
    std::cout << i << std::endl;
  }
  return 0;
}
